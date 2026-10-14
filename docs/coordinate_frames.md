# 坐标系和单位

`T_ab` 把 b 系坐标映射到 a 系：`p_a = R_ab p_b + t_ab`。齐次矩阵最后一行为 `[0,0,0,1]`。旋转为 SO(3)；四元数顺序 xyzw；长度米、角度弧度、时间仿真秒。

deck：H 中心为原点，+X 朝船首，+Y 左舷，+Z 向上；着陆面 z=0，H 涂层 z=0.025。船尾约 x=-26，船首 x=114；H 外角顺序为 (-w/2,-l/2)、(w/2,-l/2)、(w/2,l/2)、(-w/2,l/2)。

camera / OpenCV：+X 图像右、+Y 下、+Z 光轴前方。`T_deck_camera` 是相机在甲板系中的位姿，PnP 输出的 OpenCV world-to-camera 外参必须取逆。朝下的默认旋转为 diag(1,-1,-1)。右乘的小角度误差属于相机局部坐标系。

Blender 相机局部 -Z 为前方、+Y 为上；`T_world_blender_camera = T_world_deck @ T_deck_camera @ diag(1,-1,-1,1)`。相机 sensor_fit=HORIZONTAL，焦距由 fx 推导，pixel_aspect_y=fx/fy；当前 Blender 渲染仅支持零畸变。

body 为飞机 forward/right/down。`T_body_camera(t)` 由中立标定与云台 XYZ 轴角组成，平移固定，实际旋转随时间改变。编码器角速度是相机相对 body、以相机系表示；NavigationEstimate.angular_velocity_camera 是相机相对旋转 deck、以相机系表示，两者不可混用。

LiDAR：`lidar_mount=body` 时固定 `T_body_lidar`，并按测量时刻计算 `T_camera_lidar(t)=inverse(T_body_camera(t)) @ T_body_lidar`。`lidar_mount=camera` 才使用固定的 `T_camera_lidar`。两种情况均有 `T_deck_lidar=T_deck_camera @ T_camera_lidar(t)`。

ESKF 的平移/速度参考点由 filter_point_body 指定，局部旋转误差仍在相机系；输入/输出在参考点与光心之间补偿杠杆臂。对外 NavigationEstimate 的位置、速度和协方差仍统一对应光心。详见 [云台方案](gimbal_navigation.md)。

world：本项目局部海面坐标系，+Z 向上。外部舰船运动通道默认按 surge/sway/heave/roll/pitch/yaw 读取，旋转使用 Rz(yaw)Ry(pitch)Rx(roll)。若实际模型采用 NED，需要在适配 callback 中显式变换；本项目提供 ned_to_enu 辅助函数，不会猜测实际控制器约定。
