"""
ĐẠI HỌC BÁCH KHOA HÀ NỘI
TRƯỜNG ĐIỆN – ĐIỆN TỬ
Học phần: ĐỒ ÁN I - ET3290

ĐỀ TÀI: LOCALIZATION OF WIRELESS SENSOR NODES 
         WITH ERRONEOUS ANCHORS VIA EM ALGORITHM

Sinh viên thực hiện: Phan Văn Quân - 20233598
Giảng viên hướng dẫn: PGS.TS. Nguyễn Thành Chuyên

Mô tả: Mã nguồn mô phỏng Monte Carlo thuật toán 
       Expectation-Maximization (EM) cho bài toán 
       định vị WSN với mỏ neo có sai số.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
# CẤU HÌNH GIAO DIỆN THEO CHUẨN BÀI BÁO IEEE

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.linewidth'] = 1.0

# LÕI THUẬT TOÁN ĐỊNH VỊ (EM, LS, RLS)
def simulate_em_localization(sigma_a_sq, sigma_epsilon_sq=1.0, max_iter=7):
    # Khởi tạo thông số hệ thống
    N_anchors = 4
    s_true = np.array([0.0, 0.0]) # Vị trí thực của nút cảm biến
    
    # Mô phỏng tọa độ mỏ neo và nhiễu vị trí (Anchor Uncertainty)
    a_true = np.random.uniform(-10, 10, (N_anchors, 2))
    Sigma_a = sigma_a_sq * np.eye(2)
    delta_a = np.random.multivariate_normal([0, 0], Sigma_a, N_anchors)
    a_obs = a_true - delta_a # Vị trí mỏ neo quan sát được
    
    # Mô phỏng phép đo khoảng cách có chứa nhiễu (Range Measurements)
    r_obs = np.zeros(N_anchors)
    for i in range(N_anchors):
        dist_sq_true = np.sum((s_true - a_true[i])**2)
        epsilon = np.random.normal(0, np.sqrt(sigma_epsilon_sq))
        r_obs[i] = dist_sq_true + epsilon
        
    # Tuyến tính hóa hệ phương trình bằng phương pháp sai phân (Chọn mỏ neo 1 làm gốc)
    N_eq = N_anchors - 1
    H_bar = np.zeros((N_eq, 2))
    b = np.zeros(N_eq)
    
    for i in range(1, N_anchors):
        H_bar[i-1] = 2 * (a_obs[0] - a_obs[i])
        b[i-1] = r_obs[i] - r_obs[0] - np.sum(a_obs[i]**2) + np.sum(a_obs[0]**2)

    # Bộ lọc cấu hình hình học (GDOP Filter): Loại bỏ các trường hợp suy biến ma trận
    if np.linalg.cond(H_bar.T @ H_bar) > 500:
        return None, None
        
    h_bar = H_bar.flatten().reshape(2 * N_eq, 1)
    b_vec = b.reshape(N_eq, 1)

    # Xây dựng ma trận hiệp phương sai của nhiễu đo lường tổng hợp (Sigma_e)
    Sigma_e = np.zeros((N_eq, N_eq))
    for i in range(N_eq):
        for j in range(N_eq):
            if i == j:
                Sigma_e[i,j] = 2*sigma_epsilon_sq + 4*(a_obs[i+1].T @ Sigma_a @ a_obs[i+1] + a_obs[0].T @ Sigma_a @ a_obs[0])
            else:
                Sigma_e[i,j] = sigma_epsilon_sq + 4*(a_obs[0].T @ Sigma_a @ a_obs[0])
    Sigma_e_inv = np.linalg.pinv(Sigma_e)

    # Xây dựng ma trận hiệp phương sai tiên nghiệm của cấu trúc mỏ neo (Sigma_h)
    Sigma_h = np.zeros((2 * N_eq, 2 * N_eq))
    for i in range(N_eq):
        for j in range(N_eq):
            if i == j:
                Sigma_h[2*i:2*i+2, 2*j:2*j+2] = 8 * sigma_a_sq * np.eye(2)
            else:
                Sigma_h[2*i:2*i+2, 2*j:2*j+2] = 4 * sigma_a_sq * np.eye(2)
    Sigma_h_inv = np.linalg.pinv(Sigma_h)

    # Phân rã Cholesky cho Sigma_e_inv để phục vụ tính toán ma trận phạt ở Bước M
    try:
        C = scipy.linalg.cholesky(Sigma_e_inv + 1e-9 * np.eye(N_eq), lower=True)
    except scipy.linalg.LinAlgError:
        return None, None

    A_mat_LS = H_bar.T @ Sigma_e_inv @ H_bar
    
    # Ràng buộc vật lý (Physical Bound) để ngăn chặn phân kỳ nghiệm
    bound = 15.0

    # 1. Ước lượng Bình phương tối thiểu (Least Squares - LS)
    try:
        s_LS = np.linalg.pinv(A_mat_LS) @ (H_bar.T @ Sigma_e_inv @ b)
        s_LS = np.clip(s_LS, -bound, bound)
    except np.linalg.LinAlgError:
        return None, None

    # 2. Ước lượng Bình phương tối thiểu mạnh (Robust Least Squares - RLS)
    # Sử dụng điều chuẩn Tikhonov để khắc phục nhiễu Errors-In-Variables
    alpha_penalty = 0.22 * np.sqrt(sigma_a_sq)
    Sigma_H_RLS = alpha_penalty * A_mat_LS
    s_RLS = np.linalg.pinv(A_mat_LS + Sigma_H_RLS) @ (H_bar.T @ Sigma_e_inv @ b)
    s_RLS = np.clip(s_RLS, -bound, bound)

    # 3. Thuật toán Expectation-Maximization (EM)
    s_k = s_LS.copy() # Khởi tạo vòng lặp EM bằng kết quả của LS
    s_EM_history = [s_k[0]]
    
    for k in range(max_iter):
        s_k_vec = s_k.reshape(2, 1)
        I_kron_s = np.kron(np.eye(N_eq), s_k_vec.T)
        
        # Bước E (Expectation Step)
        Sigma_h_hat = np.linalg.pinv(Sigma_h_inv + I_kron_s.T @ Sigma_e_inv @ I_kron_s)
        h_hat = Sigma_h_hat @ (I_kron_s.T @ Sigma_e_inv @ b_vec + Sigma_h_inv @ h_bar)
        H_hat = h_hat.reshape(N_eq, 2)
        
        # Bước M (Maximization Step)
        sum_term_posterior = np.zeros((2, 2))
        for i in range(N_eq):
            c_i = C[:, i].reshape(N_eq, 1)
            c_i_kron_I = np.kron(c_i, np.eye(2))
            sum_term_posterior += c_i_kron_I.T @ Sigma_h_hat @ c_i_kron_I
            
        A_mat_EM = H_hat.T @ Sigma_e_inv @ H_hat + sum_term_posterior
        B_mat_EM = H_hat.T @ Sigma_e_inv @ b_vec
        
        # Cập nhật vị trí mục tiêu
        s_k = (np.linalg.pinv(A_mat_EM) @ B_mat_EM).flatten()
        s_k = np.clip(s_k, -bound, bound)
        s_EM_history.append(s_k[0])

    return s_EM_history, s_RLS[0]

# THIẾT LẬP MÔ PHỎNG MONTE CARLO
N_runs = 10000
sigma_epsilon_sq = 1.0
max_iter = 7

# Dải nhiễu khảo sát cho Fig 2
uncertainties_dB = np.arange(-10, 16, 5)

# Các mốc nhiễu tiêu biểu để khảo sát sự hội tụ (Fig 1)
fig1_db_levels = [-10, 0, 10]
results_fig1 = {db: np.zeros(max_iter + 1) for db in fig1_db_levels}

# Cấu trúc lưu trữ dữ liệu MSE cho Fig 2
results_fig2 = {'LS': [], 'RLS': [], 'EM_1': [], 'EM_2': [], 'EM_conv': []}

print(f"[*] Bắt đầu thực thi mô phỏng {N_runs} vòng lặp Monte Carlo...")

for db in uncertainties_dB:
    sigma_a_sq = 10**(-db / 10.0)
    
    errs_fig2 = {'LS': [], 'RLS': [], 'EM_1': [], 'EM_2': [], 'EM_conv': []}
    if db in fig1_db_levels:
        errs_fig1 = np.zeros(max_iter + 1)
    
    runs_completed = 0
    while runs_completed < N_runs:
        s_EM_hist, s_RLS_x = simulate_em_localization(sigma_a_sq, sigma_epsilon_sq, max_iter=max_iter)
        
        if s_EM_hist is None:
            continue
            
        # Thu thập dữ liệu đánh giá độ chính xác (Fig 2)
        errs_fig2['LS'].append((s_EM_hist[0] - 0.0)**2)
        errs_fig2['RLS'].append((s_RLS_x - 0.0)**2)
        errs_fig2['EM_1'].append((s_EM_hist[1] - 0.0)**2)
        errs_fig2['EM_2'].append((s_EM_hist[2] - 0.0)**2)
        errs_fig2['EM_conv'].append((s_EM_hist[-1] - 0.0)**2)
        
        # Thu thập dữ liệu đánh giá tốc độ hội tụ (Fig 1)
        if db in fig1_db_levels:
            errs_fig1 += np.array(s_EM_hist)**2
            
        runs_completed += 1
        
    # Tính trung bình thống kê cho Fig 2
    for k in results_fig2.keys():
        results_fig2[k].append(np.mean(errs_fig2[k]))
        
    # Tính trung bình thống kê cho Fig 1
    if db in fig1_db_levels:
        results_fig1[db] = errs_fig1 / N_runs
        
    print(f" -> Hoàn thành xử lý tại mức nhiễu: {db} dB.")
# XUẤT ĐỒ THỊ KẾT QUẢ MÔ PHỎNG

# ----------------- Fig 1: SỰ HỘI TỤ (CONVERGENCE) -----------------
plt.figure(1, figsize=(7.5, 6))
iterations = np.arange(0, max_iter + 1)
markers = ['-kv', '--ks', '-ko']

for idx, db in enumerate(fig1_db_levels):
    plt.plot(iterations, results_fig1[db], markers[idx], markerfacecolor='none',
             linewidth=1.5, markersize=7, label=f'Anchor uncertainty: $10\log(1/\sigma_a^2)$ = {db} dB')

plt.yscale('log')
plt.xlabel('Number of iterations ($k$)', fontsize=12)
plt.ylabel('MSE of estimated location on the x-axis', fontsize=12)
plt.xlim(0, max_iter)
plt.xticks(iterations)
plt.legend(loc='upper right', edgecolor='black', fancybox=False, framealpha=1.0, fontsize=10.5)
plt.title('Fig 1. Convergence of the EM algorithm.', fontsize=12, loc='left', pad=15)
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.tight_layout()

# ----------------- Fig 2: ĐỘ CHÍNH XÁC THEO NHIỄU (MSE vs UNCERTAINTY) -----------------
plt.figure(2, figsize=(7.5, 6))
plt.plot(uncertainties_dB, results_fig2['LS'], '-kv', markerfacecolor='none', linewidth=1.2, markersize=7, label='Initialization of EM(LS)')
plt.plot(uncertainties_dB, results_fig2['RLS'], '--ks', markerfacecolor='none', linewidth=1.5, markersize=7, label='RLS')
plt.plot(uncertainties_dB, results_fig2['EM_1'], '-k<', markerfacecolor='none', linewidth=1.2, markersize=7, label='EM after the $1^{st}$ iteration')
plt.plot(uncertainties_dB, results_fig2['EM_2'], '-k^', markerfacecolor='none', linewidth=1.2, markersize=7, label='EM after the $2^{nd}$ iteration')
plt.plot(uncertainties_dB, results_fig2['EM_conv'], '-k>', markerfacecolor='none', linewidth=1.2, markersize=7, label='EM after convergence')

plt.yscale('log')
plt.xlabel('Anchor uncertainty: $10\log(1/\sigma_a^2)$ (dB)', fontsize=12)
plt.ylabel('MSE of estimated location on the x-axis', fontsize=12)
plt.xlim(-10, 15)
plt.ylim(1e-2, 1e2)
plt.xticks(uncertainties_dB)
plt.legend(loc='upper right', edgecolor='black', fancybox=False, framealpha=1.0, fontsize=10.5)
plt.title('Fig 2. MSE of the estimated location v.s. the anchor uncertainty.', fontsize=12, loc='left', pad=15)
plt.tight_layout()

plt.show()