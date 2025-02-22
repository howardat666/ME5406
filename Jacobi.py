import sympy as sp

# 定义符号变量
theta, theta1, theta2 = sp.symbols('theta theta1 theta2')
tilde_theta1 = theta1 + theta
tilde_theta2 = theta2 - theta
l1 = sp.symbols('l1')  # 轮组距离
V_x, V_y, omega = sp.symbols('V_x V_y omega')  # 车体速度

# 定义雅可比矩阵 J
J = sp.Matrix([
    [sp.sin(tilde_theta1) + sp.cos(tilde_theta1), sp.cos(tilde_theta1) - sp.sin(tilde_theta1),
     l1 * (sp.cos(theta) * (sp.sin(tilde_theta1) + sp.cos(tilde_theta1)) - sp.sin(theta) * (sp.cos(tilde_theta1) - sp.sin(tilde_theta1)))],
    [sp.sin(tilde_theta1) + sp.cos(tilde_theta1), sp.cos(tilde_theta1) - sp.sin(tilde_theta1),
     -l1 * (sp.cos(theta) * (sp.sin(tilde_theta1) + sp.cos(tilde_theta1)) - sp.sin(theta) * (sp.cos(tilde_theta1) - sp.sin(tilde_theta1)))],
    [sp.sin(tilde_theta2) + sp.cos(tilde_theta2), -(sp.cos(tilde_theta2) + sp.sin(tilde_theta2)),
     l1 * (sp.cos(theta) * (sp.sin(tilde_theta2) + sp.cos(tilde_theta2)) + sp.sin(theta) * (sp.cos(tilde_theta2) + sp.sin(tilde_theta2)))],
    [sp.sin(tilde_theta2) + sp.cos(tilde_theta2), -(sp.cos(tilde_theta2) + sp.sin(tilde_theta2)),
     -l1 * (sp.cos(theta) * (sp.sin(tilde_theta2) + sp.cos(tilde_theta2)) + sp.sin(theta) * (sp.cos(tilde_theta2) + sp.sin(tilde_theta2)))]
])

# 计算广义逆矩阵 J^† = (J^T J)^(-1) J^T
J_pseudo_inverse = (J.T * J).inv() * J.T

# 展开化简结果
J_pseudo_inverse = sp.simplify(J_pseudo_inverse)
print(J_pseudo_inverse)
