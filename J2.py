import sympy as sp

# 定义变量
Vx, Vy, w = sp.symbols('Vx Vy w')  # 机器人速度
w1, w2, w3, w4 = sp.symbols('w1 w2 w3 w4')  # 轮速
theta, theta1, theta2 = sp.symbols('theta theta1 theta2')  # 方向角度
l1, l2, r = sp.symbols('l1 l2 r')  # 机器人几何参数

# 计算调整后的角度
theta1_tilde = theta1 + theta
theta2_tilde = theta2 - theta

# 计算各个轮子的速度
V1x = Vx + w * l1 * sp.cos(theta)
V1y = Vy - w * l1 * sp.sin(theta)
V2x = Vx - w * l2 * sp.cos(theta)
V2y = Vy + w * l2 * sp.sin(theta)

# 计算轮速（线速度转换）
VA = (V1x * sp.sin(theta1_tilde) + V1y * sp.cos(theta1_tilde) +
      V1x * sp.cos(theta1_tilde) - V1y * sp.sin(theta1_tilde))
VB = (V1x * sp.sin(theta1_tilde) + V1y * sp.cos(theta1_tilde) -
      V1x * sp.cos(theta1_tilde) + V1y * sp.sin(theta1_tilde))
VC = (V2x * sp.sin(theta2_tilde) - V2y * sp.cos(theta2_tilde) +
      V2x * sp.cos(theta2_tilde) + V2y * sp.sin(theta2_tilde))
VD = (V2x * sp.sin(theta2_tilde) - V2y * sp.cos(theta2_tilde) -
      V2x * sp.cos(theta2_tilde) - V2y * sp.sin(theta2_tilde))

# 轮速与机器人速度的关系
W = sp.Matrix([VA, VB, VC, VD])
V = sp.Matrix([Vx, Vy, w])

# 计算雅可比矩阵 J
J = V.jacobian(W)
J.simplify()
print(J)
