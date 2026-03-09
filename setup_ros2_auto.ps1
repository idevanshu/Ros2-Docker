Write-Host "Initializing Smart ROS 2 Humble Docker Environment..." -ForegroundColor Cyan

Write-Host "Scanning for GPU hardware..." -ForegroundColor Yellow
$gpus = Get-CimInstance Win32_VideoController
$hasNvidia = $false
$gpuNames = @()

foreach ($gpu in $gpus) {
    $gpuNames += $gpu.Name
    if ($gpu.Name -match "NVIDIA") {
        $hasNvidia = $true
    }
}

Write-Host "Detected GPUs: $($gpuNames -join ', ')" -ForegroundColor Gray

if ($hasNvidia) {
    Write-Host "NVIDIA GPU detected. Configuring for NVIDIA Container Toolkit..." -ForegroundColor Green
    $dockerfileContent = "FROM osrf/ros:humble-desktop`n`nENV NVIDIA_VISIBLE_DEVICES all`nENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute`n`nRUN apt-get update && apt-get install -y \`n    python3-pip \`n    python3-colcon-common-extensions \`n    python3-opencv \`n    nano \`n    git \`n    && rm -rf /var/lib/apt/lists/*`n`nRUN pip3 install \`n    onnxruntime-gpu \`n    numpy`n`nRUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc`nWORKDIR /ros2_ws"
} else {
    Write-Host "AMD/Intel GPU detected. Configuring for WSL 2 D3D12 Passthrough..." -ForegroundColor Blue
    $dockerfileContent = "FROM osrf/ros:humble-desktop`n`nRUN apt-get update && apt-get install -y \`n    python3-pip \`n    python3-colcon-common-extensions \`n    python3-opencv \`n    nano \`n    git \`n    mesa-utils \`n    libgl1-mesa-dri \`n    libglx-mesa0 \`n    && rm -rf /var/lib/apt/lists/*`n`nRUN pip3 install \`n    onnxruntime \`n    numpy`n`nRUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc`nWORKDIR /ros2_ws"
}
Set-Content -Path Dockerfile -Value $dockerfileContent
Write-Host "Created tailored Dockerfile" -ForegroundColor Green

if ($hasNvidia) {
    $composeContent = "version: '3.8'`n`nservices:`n  ros2_dev:`n    build: .`n    container_name: ros2_auto_container`n    stdin_open: true`n    tty: true`n    network_mode: host`n    environment:`n      - DISPLAY=:0`n      - QT_X11_NO_MITSHM=1`n    volumes:`n      - /tmp/.X11-unix:/tmp/.X11-unix:rw`n      - ./workspace:/ros2_ws`n    deploy:`n      resources:`n        reservations:`n          devices:`n            - driver: nvidia`n              count: 1`n              capabilities: [gpu, compute, graphics, utility]"
} else {
    $composeContent = "version: '3.8'`n`nservices:`n  ros2_dev:`n    build: .`n    container_name: ros2_auto_container`n    stdin_open: true`n    tty: true`n    network_mode: host`n    environment:`n      - DISPLAY=:0`n      - QT_X11_NO_MITSHM=1`n      - GALLIUM_DRIVER=d3d12`n      - LIBGL_ALWAYS_SOFTWARE=0`n      - LD_LIBRARY_PATH=/usr/lib/wsl/lib`n    volumes:`n      - /tmp/.X11-unix:/tmp/.X11-unix:rw`n      - /usr/lib/wsl:/usr/lib/wsl:ro`n      - ./workspace:/ros2_ws`n    devices:`n      - /dev/dri:/dev/dri`n      - /dev/dxg:/dev/dxg"
}
Set-Content -Path docker-compose.yml -Value $composeContent
Write-Host "Created tailored docker-compose.yml" -ForegroundColor Green

# Create workspace directory
if (-Not (Test-Path "workspace")) {
    New-Item -ItemType Directory -Path "workspace" | Out-Null
    Write-Host "Created local workspace directory" -ForegroundColor Green
}

# Create YOLOv8 ROS2 package structure
$pkgPath = "workspace/src/yolo_detector/yolo_detector"
if (-Not (Test-Path $pkgPath)) {
    New-Item -ItemType Directory -Path $pkgPath -Force | Out-Null
    Write-Host "Created yolo_detector package structure" -ForegroundColor Green
}

# __init__.py
Set-Content -Path "$pkgPath/__init__.py" -Value ""

# yolo_node.py
$yoloNode = "import rclpy`nfrom rclpy.node import Node`nfrom sensor_msgs.msg import Image`nfrom cv_bridge import CvBridge`nimport cv2`nimport numpy as np`nimport onnxruntime as ort`nimport os`n`nclass YoloNode(Node):`n    def __init__(self):`n        super().__init__('yolo_detector')`n        self.bridge = CvBridge()`n        model_path = '/ros2_ws/yolov8n.onnx'`n        if not os.path.exists(model_path):`n            self.get_logger().error(f'Model not found at {model_path}. Place yolov8n.onnx in your workspace/ folder.')`n            return`n        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']`n        self.session = ort.InferenceSession(model_path, providers=providers)`n        self.input_name = self.session.get_inputs()[0].name`n        used_provider = self.session.get_providers()[0]`n        self.get_logger().info(f'YOLOv8n loaded. Running on: {used_provider}')`n        self.sub = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)`n        self.pub = self.create_publisher(Image, '/yolo/detection_image', 10)`n`n    def preprocess(self, frame):`n        img = cv2.resize(frame, (640, 640))`n        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0`n        return np.ascontiguousarray(img)[np.newaxis]`n`n    def callback(self, msg):`n        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')`n        h, w = frame.shape[:2]`n        input_tensor = self.preprocess(frame)`n        outputs = self.session.run(None, {self.input_name: input_tensor})`n        detections = outputs[0][0].T`n        for det in detections:`n            cx, cy, bw, bh = det[:4]`n            scores = det[4:]`n            conf = float(scores.max())`n            cls = int(scores.argmax())`n            if conf > 0.5:`n                x1 = int((cx - bw/2) * w / 640)`n                y1 = int((cy - bh/2) * h / 640)`n                x2 = int((cx + bw/2) * w / 640)`n                y2 = int((cy + bh/2) * h / 640)`n                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)`n                cv2.putText(frame, f'cls:{cls} {conf:.2f}', (x1, y1-5),`n                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)`n        out_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')`n        self.pub.publish(out_msg)`n`ndef main(args=None):`n    rclpy.init(args=args)`n    node = YoloNode()`n    rclpy.spin(node)`n    rclpy.shutdown()"
Set-Content -Path "$pkgPath/yolo_node.py" -Value $yoloNode

# setup.py
$setupPy = "from setuptools import setup`n`nsetup(`n    name='yolo_detector',`n    version='0.1.0',`n    packages=['yolo_detector'],`n    install_requires=['setuptools'],`n    entry_points={`n        'console_scripts': [`n            'yolo_node = yolo_detector.yolo_node:main',`n        ],`n    },`n)"
Set-Content -Path "workspace/src/yolo_detector/setup.py" -Value $setupPy

# package.xml
$packageXml = "<?xml version='1.0'?>`n<package format='3'>`n  <name>yolo_detector</name>`n  <version>0.1.0</version>`n  <description>YOLOv8n ONNX detector node for ROS 2</description>`n  <maintainer email='you@example.com'>you</maintainer>`n  <license>MIT</license>`n  <depend>rclpy</depend>`n  <depend>sensor_msgs</depend>`n  <depend>cv_bridge</depend>`n</package>"
Set-Content -Path "workspace/src/yolo_detector/package.xml" -Value $packageXml

Write-Host "Created YOLOv8 detector ROS2 package" -ForegroundColor Green

Write-Host "Building the Docker image..." -ForegroundColor Yellow
docker compose build

Write-Host "Starting the container..." -ForegroundColor Yellow
docker compose up -d

Write-Host "Setup Complete! Attach to your container using:" -ForegroundColor Green
Write-Host "docker exec -it ros2_auto_container bash" -ForegroundColor Cyan
Write-Host "" 
Write-Host "--- YOLOv8 Next Steps ---" -ForegroundColor Yellow
Write-Host "1. Place yolov8n.onnx inside the workspace/ folder" -ForegroundColor White
Write-Host "2. Inside container run: cd /ros2_ws && colcon build --packages-select yolo_detector" -ForegroundColor White
Write-Host "3. Then run: source install/setup.bash && ros2 run yolo_detector yolo_node" -ForegroundColor White
