# MPC and Policy Tuning Guide

Tai lieu nay giai thich y nghia tung tham so trong file mpc_tuning.yaml va cach tuning thuc te.

## 1) Tong quan tan so

- Tan so MPC:
  - f_mpc = 1 / mpc.dt
  - Vi du: mpc.dt = 0.25 -> 4 Hz
- Tan so policy loop:
  - f_policy = 1 / timer_period
  - Vi du: timer_period = 0.1 -> 10 Hz

Luu y:
- Policy loop la tan so gui request den /ocp_plann (muc tieu).
- Tan so thuc te co the thap hon neu service xu ly lau.

## 2) Nhom tham so opt_planner.ros__parameters

### mpc.horizon_steps
- Y nghia: So buoc du doan trong horizon MPC.
- Tac dong:
  - Tang horizon -> nhin xa hon, plan on dinh hon trong tinh huong phuc tap.
  - Qua lon -> tinh toan nang hon, de tre.
- Huong tuning:
  - Bat dau: 8-12.
  - Neu robot hay dao dong truoc nga re/chuong ngai: tang nhe (+1, +2).
  - Neu CPU cao, response cham: giam nhe.
- Ghi chu quan trong:
  - Hien tai code clamp horizon toi da bang kNP compile-time (=10).

### mpc.dt
- Y nghia: Buoc thoi gian roi rac cua mo hinh MPC (giay).
- Tac dong:
  - Nho hon -> phan giai thoi gian min hon, dieu khien sat hon, nhung nang tinh toan.
  - Lon hon -> nhe tinh toan, nhung tho hon.
- Huong tuning:
  - Thuong dung: 0.1-0.3.
  - He can phan ung nhanh: giam dt (0.25 -> 0.2 hoac 0.15).
  - He bi qua tai CPU: tang dt.

### mpc.max_linear_vel
- Y nghia: Gioi han van toc tuyen tinh trong rang buoc banh.
- Tac dong:
  - Nho -> an toan hon, robot cham hon.
  - Lon -> nhanh hon nhung de vuot kha nang co khi.
- Huong tuning:
  - Dat theo kha nang thuc cua de va he truyen dong.

### mpc.max_angular_vel
- Y nghia: Gioi han van toc goc toi da.
- Tac dong:
  - Nho -> quay mem, it giat.
  - Lon -> quay nhanh, de tao chuyen dong gap neu qua cao.
- Huong tuning:
  - Dat theo dinh muc robot, giam neu thay quay giat.

### mpc.max_linear_acc
- Y nghia: Gioi han gia toc dieu khien (thong qua rang buoc input).
- Tac dong:
  - Nho -> em hon, it giat.
  - Lon -> bam muc tieu nhanh hon nhung de giat.
- Huong tuning:
  - Neu robot giat khi xuat phat/dung: giam.
  - Neu robot phan ung tre: tang nhe.

### mpc.max_angular_acc
- Y nghia: Gioi han gia toc goc toi da.
- Tac dong:
  - Nho -> quay muot.
  - Lon -> doi huong nhanh hon.
- Huong tuning:
  - Dieu chinh cung cap voi max_angular_vel de tranh quay cuc doan.

### mpc.wheel_half_track
- Y nghia: Nua khoang cach 2 banh, dung trong chuyen doi v/w <-> banh trai/phai.
- Tac dong:
  - Sai tham so nay se lam mapping control sai, robot di khong dung y.
- Huong tuning:
  - Khong tuning theo hieu nang; dat dung thong so hinh hoc that cua robot.

### mpc.local_obst_num
- Y nghia: So chuong ngai gan nhat dua vao bai toan MPC.
- Tac dong:
  - Lon -> an toan hon trong moi truong dong, nhung nang tinh toan.
  - Nho -> nhe tinh toan, nhung de bo sot vat can.
- Huong tuning:
  - Bat dau: 6-10.
  - Tang neu hay bo sot va cham trong dam dong.

## 3) Nhom trong so MPC (mpc.weights.*)

Muc tieu tong quat:
- Tang trong so loi -> bam quyd ao/muc tieu manh hon.
- Tang trong so input/smooth -> dieu khien mem hon.

### mpc.weights.slack
- Y nghia: Phat bien slack cho cac rang buoc tranh va cham.
- Tac dong:
  - Rat lon -> uu tien an toan, tranh vi pham rang buoc.
  - Qua lon co the lam bai toan cung/ kho hoi tu trong tinh huong chat.
- Huong tuning:
  - Giu lon (vi du 1e4-1e5).
  - Neu solver hay fail do qua cung: giam nhe.

### mpc.weights.pose_x, mpc.weights.pose_y
- Y nghia: Phat sai so bam quyd ao theo truc x/y.
- Tac dong:
  - Tang -> bam sat duong hon.
  - Qua cao -> robot de giat khi gap duong cong/nhieu vat can.
- Huong tuning:
  - Tang khi robot bi lech quyd ao.
  - Giam neu dao dong.

### mpc.weights.yaw_rate
- Y nghia: Phat toc do quay (r) de han che quay manh.
- Tac dong:
  - Tang -> it quay gap, muot hon.
  - Giam -> linh hoat quay nhanh hon.
- Huong tuning:
  - Neu robot lac lau khi quay: tang.

### mpc.weights.terminal_x, mpc.weights.terminal_y
- Y nghia: Phat loi tai diem cuoi horizon (terminal cost).
- Tac dong:
  - Tang -> MPC nhin den diem cuoi horizon manh hon, giam lanh quanh.
  - Qua cao -> de gay hanh vi nong, "lao" den diem cuoi.
- Huong tuning:
  - Tang vua phai neu robot cham vao duong cua khuc sau.

### mpc.weights.input_acc
- Y nghia: Phat do lon lenh gia toc thang.
- Tac dong:
  - Tang -> giam lenh gia toc lon, di em hon.
  - Giam -> robot xung manh hon.
- Huong tuning:
  - Neu xuat hien giat toc do: tang.

### mpc.weights.input_yaw_acc
- Y nghia: Phat do lon lenh gia toc quay.
- Tac dong:
  - Tang -> quay em hon.
  - Giam -> quay nhanh hon.
- Huong tuning:
  - Neu quay giat: tang.

### mpc.weights.smooth_acc
- Y nghia: Phat do thay doi gia toc thang giua 2 buoc lien tiep.
- Tac dong:
  - Tang -> profile van toc muot hon.
  - Qua cao -> robot phan ung cham.
- Huong tuning:
  - Tang neu jerk theo truc thang cao.

### mpc.weights.smooth_yaw_acc
- Y nghia: Phat do thay doi gia toc quay giua 2 buoc lien tiep.
- Tac dong:
  - Tang -> quay muot, giam giat huong.
  - Qua cao -> doi huong cham.
- Huong tuning:
  - Tang neu robot dao dau khi re.

## 4) Nhom tham so rl_ocp_policy_bridge.ros__parameters

### timer_period
- Y nghia: Chu ky vong policy/control loop (giay).
- Tac dong:
  - Nho -> loop nhanh hon (Hz cao), phan ung nhanh hon, tang tai CPU.
  - Lon -> loop cham hon, de tre tang.
- Huong tuning:
  - Bat dau: 0.1 (10 Hz).
  - Robot phan ung cham: giam xuong 0.08 hoac 0.05 neu may du suc.
  - CPU cao: tang len 0.12-0.2.

### control_dt
- Y nghia: Buoc tich phan de doi gia toc planner (al/ar) sang lenh toc do gui cmd_vel.
- Tac dong:
  - Lon -> buoc cap nhat toc do moi lan lon hon (de giat).
  - Nho -> cap nhat mem hon.
- Huong tuning:
  - Nen dong bo voi mpc.dt de hanh vi on dinh (thuong de bang nhau).

## 5) Quy trinh tuning de xuat (thuc chien)

1. Co dinh thong so hinh hoc dung truoc:
- mpc.wheel_half_track

2. Chon nhip he thong:
- Chon mpc.dt theo tai tinh toan va yeu cau phan ung.
- Chon timer_period (thuong nhanh hon hoac bang nhip MPC service mong muon).

3. Chon horizon:
- Boi canh trong nha/phong nho: 8-10 la du.
- Boi canh mo, can nhin xa: tang horizon (neu sau nay mo rong kNP).

4. Tuning an toan va muot:
- Tang slack neu can uu tien tranh va cham.
- Tang input_* va smooth_* neu robot giat.

5. Tuning bam quyd ao:
- Tang pose_x/pose_y va terminal_x/terminal_y neu lech duong.
- Neu dao dong, giam nhe pose_* hoac tang smooth_*.

6. Kiem tra KPI moi lan doi tham so:
- Ti le success cua /ocp_plann
- Thoi gian solve trung binh
- Do lech quyd ao
- Muc do giat (jerk) khi tang/ giam toc va khi quay

## 6) Mau tuning nhanh theo hien tuong

- Hien tuong: Robot giat khi re
  - Tang mpc.weights.input_yaw_acc
  - Tang mpc.weights.smooth_yaw_acc
  - Giam nhe mpc.max_angular_acc

- Hien tuong: Robot cham, duoi muc tieu
  - Giam mpc.weights.input_acc
  - Giam mpc.weights.smooth_acc
  - Giam mpc.dt (neu CPU cho phep)

- Hien tuong: Hay lech duong tham chieu
  - Tang mpc.weights.pose_x, mpc.weights.pose_y
  - Tang mpc.weights.terminal_x, mpc.weights.terminal_y

- Hien tuong: Hay vi pham an toan gan vat can
  - Tang mpc.weights.slack
  - Tang mpc.local_obst_num
  - Giam mpc.max_linear_vel

## 7) Ghi chu van hanh

- Moi lan doi tham so, nen thay doi tung nhom nho (1-2 tham so), test lai, ghi log.
- Tranh thay doi nhieu tham so cung luc vi kho truy vet nguyen nhan.
- Neu doi file config ma launch dang dung install space cu, hay rebuild/sourcing lai workspace de dam bao node nap dung file moi.
