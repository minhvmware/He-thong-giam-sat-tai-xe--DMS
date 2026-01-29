import cv2
import numpy as np
import mediapipe as mp
import time
import math
import threading
import tkinter as tk
from tkinter import ttk
from playsound import playsound

#chinh sua du lieu o day!
nguong_mat = 0.2
so_frame_mat = 15
nguong_mieng = 1.25
nguong_ngua = 20.0
nguong_quay = 30.0
FILE_AM_THANH = 'chiken-on-tree.mp3'

# thong tin mat
mat_phai = (33, 160, 158, 133, 153, 144)
mat_trai = (362, 385, 387, 263, 373, 380)
chi_so_mieng = (61, 39, 0, 269, 291, 405, 17, 181)
chi_so_pose = (1, 152, 33, 263, 61, 291)

# 3D Model
mo_hinh_3d = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

# gia tri dat tam
dem_mat = 0
thoi_gian_tay = None
thoi_gian_fps_truoc = time.time()
danh_sach_fps = []
last_sound_time = 0
COOLDOWN_AM_THANH = 3.0

# Filters
bo_loc_mat = {'x_y': None, 'dx_y': None, 'last_t': None}
bo_loc_mieng = {'x_y': None, 'dx_y': None, 'last_t': None}
bo_loc_pose = [{'x_y': None, 'dx_y': None, 'last_t': None} for _ in range(3)]

# camera o day
def hien_cua_so_chon_camera():
    root = tk.Tk()
    root.title("Cài đặt Camera")
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (300/2)
    y = (hs/2) - (150/2)
    root.geometry('%dx%d+%d+%d' % (300, 150, x, y))

    ket_qua = [0]
    danh_sach_cam = []
    
    lbl_status = tk.Label(root, text="Đang quét camera...", fg="blue")
    lbl_status.pack(pady=5)
    root.update()

    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            danh_sach_cam.append(f"Camera {i}")
            cap.release()
    
    lbl_status.config(text="Chọn Camera của bạn:")
    if not danh_sach_cam:
        lbl_status.config(text="Không tìm thấy Camera!", fg="red")
        danh_sach_cam.append("Không có camera")

    combo = ttk.Combobox(root, values=danh_sach_cam, state="readonly")
    if danh_sach_cam: combo.current(0)
    combo.pack(pady=10)

    def xac_nhan():
        lua_chon = combo.get()
        if "Camera" in lua_chon:
            idx = int(lua_chon.split(" ")[1])
            ket_qua[0] = idx
        root.destroy()

    btn = tk.Button(root, text="BẮT ĐẦU CHẠY", command=xac_nhan, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
    btn.pack(pady=10)
    root.mainloop()
    return ket_qua[0]

# ham can xu ly
def play_sound_async():
    try: playsound(FILE_AM_THANH, block=False)
    except: pass

def kich_hoat_canh_bao():
    global last_sound_time
    hien_tai = time.time()
    if hien_tai - last_sound_time > COOLDOWN_AM_THANH:
        last_sound_time = hien_tai
        t = threading.Thread(target=play_sound_async)
        t.start()

def tinh_ti_le_mat(cac_diem_mat):
    if len(cac_diem_mat) != 6: return 0.0
    a = np.linalg.norm(cac_diem_mat[1] - cac_diem_mat[5])
    b = np.linalg.norm(cac_diem_mat[2] - cac_diem_mat[4])
    c = np.linalg.norm(cac_diem_mat[0] - cac_diem_mat[3])
    return (a + b) / (2.0 * c) if c != 0 else 0.0

def tinh_ti_le_mieng(cac_diem_mieng):
    if len(cac_diem_mieng) != 8: return 0.0
    a = np.linalg.norm(cac_diem_mieng[1] - cac_diem_mieng[7])
    b = np.linalg.norm(cac_diem_mieng[2] - cac_diem_mieng[6])
    c = np.linalg.norm(cac_diem_mieng[3] - cac_diem_mieng[5])
    d = np.linalg.norm(cac_diem_mieng[0] - cac_diem_mieng[4])
    return (a + b + c) / (2.0 * d) if d != 0 else 0.0

def bo_loc_euro(gia_tri, thoi_gian, du_lieu_loc, min_cut=1.0, beta=0.007, d_cut=1.0):
    if du_lieu_loc['last_t'] is None:
        du_lieu_loc['last_t'] = thoi_gian
        du_lieu_loc['x_y'] = gia_tri
        du_lieu_loc['dx_y'] = 0.0
        return gia_tri
    khoang_thoi_gian = max(thoi_gian - du_lieu_loc['last_t'], 1.0/30.0)
    du_lieu_loc['last_t'] = thoi_gian
    dao_ham = (gia_tri - du_lieu_loc['x_y']) / khoang_thoi_gian
    tau_d = 1.0 / (2.0 * 3.14159 * d_cut)
    alpha_d = 1.0 / (1.0 + tau_d / khoang_thoi_gian)
    du_lieu_loc['dx_y'] = alpha_d * dao_ham + (1 - alpha_d) * du_lieu_loc['dx_y']
    cutoff = min_cut + beta * abs(du_lieu_loc['dx_y'])
    tau = 1.0 / (2.0 * 3.14159 * cutoff)
    alpha = 1.0 / (1.0 + tau / khoang_thoi_gian)
    du_lieu_loc['x_y'] = alpha * gia_tri + (1 - alpha) * du_lieu_loc['x_y']
    return du_lieu_loc['x_y']

def lay_huong_dau(diem_mat, chieu_rong, chieu_cao):
    cam = np.array([[chieu_rong, 0, chieu_rong/2], [0, chieu_rong, chieu_cao/2], [0, 0, 1]], dtype=np.float64)
    cac_diem = np.array([[diem_mat[i].x*chieu_rong, diem_mat[i].y*chieu_cao] for i in chi_so_pose], dtype=np.float64)
    thanh_cong, vec_quay, vec_dich = cv2.solvePnP(mo_hinh_3d, cac_diem, cam, np.zeros((4,1)))
    if not thanh_cong: return 0, 0, 0, None, None
    ma_tran_quay, _ = cv2.Rodrigues(vec_quay)
    sy = np.sqrt(ma_tran_quay[0,0]**2 + ma_tran_quay[1,0]**2)
    if sy > 1e-6:
        ngua = np.degrees(np.arctan2(-ma_tran_quay[2,0], sy))
        quay = np.degrees(np.arctan2(ma_tran_quay[1,0], ma_tran_quay[0,0]))
        nghieng = np.degrees(np.arctan2(ma_tran_quay[2,1], ma_tran_quay[2,2]))
    else:
        ngua, quay, nghieng = np.degrees(np.arctan2(-ma_tran_quay[2,0], sy)), 0, np.degrees(np.arctan2(-ma_tran_quay[1,2], ma_tran_quay[1,1]))
    return ngua, quay, nghieng, vec_quay, vec_dich

def lam_sang_anh(khung_hinh):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ycrcb = cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2YCrCb)
    cac_kenh = list(cv2.split(ycrcb))
    cac_kenh[0] = clahe.apply(cac_kenh[0])
    return cv2.cvtColor(cv2.merge(cac_kenh), cv2.COLOR_YCrCb2BGR)

def ve_len_man_hinh(khung_hinh, du_lieu_mat, du_lieu_tay, fps_hien_tai, danh_sach_canh_bao):
    cao, rong = khung_hinh.shape[:2]
    # Panel
    cv2.rectangle(khung_hinh, (10, 10), (200, 140), (0,0,0), -1)
    cv2.rectangle(khung_hinh, (10, 10), (200, 140), (255,255,255), 1)
    
    # Info
    mau_mat = (0, 0, 255) if du_lieu_mat.get('ti_le_mat', 0) < nguong_mat else (0, 255, 0)
    mau_mieng = (0, 0, 255) if du_lieu_mat.get('ti_le_mieng', 0) > nguong_mieng else (0, 255, 0)
    
    cv2.putText(khung_hinh, f"EAR: {du_lieu_mat.get('ti_le_mat', 0):.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mau_mat, 1)
    cv2.putText(khung_hinh, f"MAR: {du_lieu_mat.get('ti_le_mieng',  0):.2f}", (20, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mau_mieng, 1)
    
    # --- ĐÃ THÊM LẠI PHẦN BỊ THIẾU Ở ĐÂY ---
    cv2.putText(khung_hinh, f"Pitch: {du_lieu_mat.get('goc_ngua', 0):.1f}", (20, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(khung_hinh, f"Yaw: {du_lieu_mat.get('goc_quay', 0):.1f}", (20, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(khung_hinh, f"Roll: {du_lieu_mat.get('goc_nghieng', 0):.1f}", (20, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    # ----------------------------------------
    
    cv2.putText(khung_hinh, f"FPS: {fps_hien_tai:.1f}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)

    # Ve mat
    if du_lieu_mat.get('diem_mat'):
        diem = du_lieu_mat['diem_mat']
        for idx in mat_phai + mat_trai + chi_so_mieng:
            cv2.circle(khung_hinh, (int(diem[idx].x*rong), int(diem[idx].y*cao)), 1, (0,255,0), -1)
    
    # Ve truc dau
    if du_lieu_mat.get('vec_quay') is not None and du_lieu_mat.get('diem_mat'):
        cam = np.array([[rong,0,rong/2],[0,rong,cao/2],[0,0,1]], dtype=np.float64)
        truc = np.float64([[100,0,0],[0,100,0],[0,0,100]])
        diem_chieu, _ = cv2.projectPoints(truc, du_lieu_mat['vec_quay'], du_lieu_mat['vec_dich'], cam, np.zeros((4,1)))
        mui = (int(du_lieu_mat['diem_mat'][1].x*rong), int(du_lieu_mat['diem_mat'][1].y*cao))
        try:
            cv2.line(khung_hinh, mui, tuple(diem_chieu[0].ravel().astype(int)), (0,0,255), 2)
            cv2.line(khung_hinh, mui, tuple(diem_chieu[1].ravel().astype(int)), (0,255,0), 2)
            cv2.line(khung_hinh, mui, tuple(diem_chieu[2].ravel().astype(int)), (255,0,0), 2)
        except: pass

    # Ve tay
    if du_lieu_tay.get('danh_sach_tay'):
        ve_tay = mp.solutions.drawing_utils
        kieu_tay = mp.solutions.hands
        for tay in du_lieu_tay['danh_sach_tay']:
            ve_tay.draw_landmarks(khung_hinh, tay, kieu_tay.HAND_CONNECTIONS, ve_tay.DrawingSpec(color=(0,255,0), thickness=1), ve_tay.DrawingSpec(color=(255,255,255), thickness=1))
    
    # Ve Canh Bao
    for i, noi_dung in enumerate(danh_sach_canh_bao):
        y = cao - 30 - i*35
        kich_thuoc = cv2.getTextSize(noi_dung, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x = (rong - kich_thuoc[0]) // 2
        cv2.rectangle(khung_hinh, (x-10, y-25), (x+kich_thuoc[0]+10, y+5), (0,0,0), -1)
        cv2.putText(khung_hinh, noi_dung, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    return khung_hinh

def chay_chuong_trinh():
    global dem_mat, thoi_gian_tay, thoi_gian_fps_truoc, danh_sach_fps
    
    id_camera = hien_cua_so_chon_camera()
    print(f"Đã chọn Camera ID: {id_camera}")

    luoi_mat = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    theo_doi_tay = mp.solutions.hands.Hands(max_num_hands=2)
    
    camera = cv2.VideoCapture(id_camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not camera.isOpened():
        print(f"Loi: Khong mo duoc camera so {id_camera}!")
        return
    
    while True:
        doc_duoc, khung_hinh = camera.read()
        if not doc_duoc: break
        
        thoi_gian_hien_tai = time.time()
        cao, rong = khung_hinh.shape[:2]
        khung_hinh = lam_sang_anh(khung_hinh)
        anh_rgb = cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2RGB)
        
        # Xu ly Mat
        ket_qua_mat = luoi_mat.process(anh_rgb)
        du_lieu_mat = {'buon_ngu': False, 'dang_ngap': False, 'canh_bao_dau': False}
        if ket_qua_mat.multi_face_landmarks:
            diem_mat = ket_qua_mat.multi_face_landmarks[0].landmark
            du_lieu_mat['diem_mat'] = diem_mat
            tat_ca_x = [d.x for d in diem_mat]
            tat_ca_y = [d.y for d in diem_mat]
            du_lieu_mat['khung_mat'] = {'x_min': min(tat_ca_x), 'x_max': max(tat_ca_x), 'y_min': min(tat_ca_y), 'y_max': max(tat_ca_y)}
            
            diem_mat_phai = [np.array([diem_mat[i].x*rong, diem_mat[i].y*cao]) for i in mat_phai]
            diem_mat_trai = [np.array([diem_mat[i].x*rong, diem_mat[i].y*cao]) for i in mat_trai]
            ti_le_mat_tho = (tinh_ti_le_mat(diem_mat_phai) + tinh_ti_le_mat(diem_mat_trai)) / 2
            du_lieu_mat['ti_le_mat'] = bo_loc_euro(ti_le_mat_tho, thoi_gian_hien_tai, bo_loc_mat)
            
            if du_lieu_mat['ti_le_mat'] < nguong_mat:
                dem_mat += 1
                if dem_mat >= so_frame_mat: du_lieu_mat['buon_ngu'] = True
            else: dem_mat = 0

            diem_mieng = [np.array([diem_mat[i].x*rong, diem_mat[i].y*cao]) for i in chi_so_mieng]
            ti_le_mieng_tho = tinh_ti_le_mieng(diem_mieng)
            du_lieu_mat['ti_le_mieng'] = bo_loc_euro(ti_le_mieng_tho, thoi_gian_hien_tai, bo_loc_mieng)
            du_lieu_mat['dang_ngap'] = du_lieu_mat['ti_le_mieng'] > nguong_mieng
            
            goc_ngua, goc_quay, goc_nghieng, vec_quay, vec_dich = lay_huong_dau(diem_mat, rong, cao)
            du_lieu_mat['goc_ngua'] = bo_loc_euro(goc_ngua, thoi_gian_hien_tai, bo_loc_pose[0])
            du_lieu_mat['goc_quay'] = bo_loc_euro(goc_quay, thoi_gian_hien_tai, bo_loc_pose[1])
            du_lieu_mat['goc_nghieng'] = bo_loc_euro(goc_nghieng, thoi_gian_hien_tai, bo_loc_pose[2])
            du_lieu_mat['vec_quay'], du_lieu_mat['vec_dich'] = vec_quay, vec_dich
            du_lieu_mat['canh_bao_dau'] = abs(du_lieu_mat['goc_ngua']) > nguong_ngua or abs(du_lieu_mat['goc_quay']) > nguong_quay
        else: dem_mat = 0

        # Xu ly Tay
        ket_qua_tay = theo_doi_tay.process(anh_rgb)
        du_lieu_tay = {'mat_tap_trung': False, 'danh_sach_tay': []}
        if ket_qua_tay.multi_hand_landmarks:
            du_lieu_tay['danh_sach_tay'] = ket_qua_tay.multi_hand_landmarks
            if du_lieu_mat.get('khung_mat'):
                khung = du_lieu_mat['khung_mat']
                co_tay_gan = False
                for tay in ket_qua_tay.multi_hand_landmarks:
                    tam_x, tam_y = (tay.landmark[0].x + tay.landmark[9].x)/2, (tay.landmark[0].y + tay.landmark[9].y)/2
                    w, h = khung['x_max'] - khung['x_min'], khung['y_max'] - khung['y_min']
                    if (khung['x_min']-w*0.2 <= tam_x <= khung['x_max']+w*0.2 and khung['y_min']-h*0.2 <= tam_y <= khung['y_max']+h*0.2):
                        co_tay_gan = True; break
                if co_tay_gan:
                    if not thoi_gian_tay: thoi_gian_tay = time.time()
                    elif time.time() - thoi_gian_tay >= 3.0: du_lieu_tay['mat_tap_trung'] = True
                else: thoi_gian_tay = None
        else: thoi_gian_tay = None

        # Canh bao
        danh_sach_canh_bao = []
        if du_lieu_mat.get('buon_ngu'): danh_sach_canh_bao.append("CANH BAO BUON NGU!")
        if du_lieu_mat.get('dang_ngap'): danh_sach_canh_bao.append("PHAT HIEN NGAP")
        if du_lieu_mat.get('canh_bao_dau'): danh_sach_canh_bao.append("NHIN VE PHIA TRUOC!")
        if du_lieu_tay.get('mat_tap_trung'): danh_sach_canh_bao.append("CANH BAO MAT TAP TRUNG!")
        if danh_sach_canh_bao: kich_hoat_canh_bao()

        # Render
        danh_sach_fps.append(1.0 / max(time.time() - thoi_gian_fps_truoc, 0.0001))
        thoi_gian_fps_truoc = time.time()
        if len(danh_sach_fps) > 30: danh_sach_fps.pop(0)
        khung_hinh = ve_len_man_hinh(khung_hinh, du_lieu_mat, du_lieu_tay, sum(danh_sach_fps)/len(danh_sach_fps), danh_sach_canh_bao)
        
        cv2.imshow("He Thong Giam Sat Tai Xe", khung_hinh)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    luoi_mat.close(); theo_doi_tay.close(); camera.release(); cv2.destroyAllWindows()
    print("Da tat chuong trinh")

if __name__ == "__main__":
    chay_chuong_trinh()