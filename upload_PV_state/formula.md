Plese refer to Wei Zhang's dissertation Chapter 2 for details.

Plasma velocity tangential to the magnetopause surface
$\mathbf{u}_t = \mathbf{u} - (\mathbf{u} \cdot \hat{\mathbf{n}})\hat{\mathbf{n}}$\\

1. Contour. Have the MP surface $B_z=0$, the normal direction is calculated automaticcaly (Normals). 
2. Calculator. U_vec = Ux * iHat + Uy * jHat + Uz * kHat
3. Calculator. UdotN = dot(U_vec, Normals)
4. Calculator. U_t = U_vec - UdotN * Normals

Similarly, we can obtain the plasma velocity perpendicular to the local magnetic field.
$U_{perp} = \frac{|\mathbf{u} \times \mathbf{B}|}{|\mathbf{B}|}$ \\

1. Calculator. U_vec = Ux * iHat + Uy * jHat + Uz * kHat
2. Calculator. B_vec = Bx * iHat + By * jHat + Bz * kHat
3. Calculator. B_mag = sqrt(Bx^2 + By^2 + Bz^2)
4. Calculator. U_perp = U_vec - dot(U_vec, B_vec) * B_vec / B_mag ^ 2

The magnetic tension forces $\mathbf{F}_{tension} = \frac{1}{\mu_0}(\mathbf{B} \cdot \nabla)\mathbf{B}$ \\

1. Gradient. get GradBx = Gradient(Bx) (## results is 3 compoents GradBx_x, GradBx_y, GradBx_z##)
2. Gradient. get GradBy = Gradient(By)
3. Gradient. get GradBz = Gradient(Bz)
4. Calculator. Get Tension force: F_tension_x = Bx * GradBx_x + By * GradBx_y + Bz * GradBx_z
5. Calculator. F_tension_y = Bx * GradBy_x + By * GradBy_y + Bz * GradBy_z
6. Calculator. F_tension_z = Bx * GradBz_x + By * GradBz_y + Bz * GradBz_z
7. Calculator. Normalization. F_tension = (F_tension_x * iHat + F_tension_y * jHat + F_tension_z * kHat) / 6.371 /4/np.pi/100


JxB force is easy in the ideal MHD simulation \\

1. Calculator. J_vec = Jx * iHat + Jy * jHat + Jz * kHat
2. Calculator. B_vec = Bx * iHat + By * jHat + Bz * kHat
3. Calculator. JxB = cross(J_vec, B_vec) 

In the MHD-AEPIC, please know the mi/me mass ratio: s and the scaling factor: scaling
1. Calculator. Ui_vec = Uix * iHat + Uiy * jHat + Uiz*kHat
2. Calculator. Ue_vec = Uex * iHat + Uey * jHat + Uez*kHat
3. Calculator. J_vec = (rhoi * Ui_vec - rhoe * Ue_vec * s ) * 1e15 * 1.6 * 10^-19 / scaling
4. Calculator. B_vec = Bx * iHat + By * jHat + Bz * kHat
5. Calculator. JxB = cross(J_vec, B_vec) 
