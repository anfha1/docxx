# ĐỐI CHIẾU 5 YÊU CẦU ĐỀ BÀI VÀ THỰC TẾ TRIỂN KHAI

Dựa trên mã nguồn và toàn bộ dự án bạn đã thực hiện trong thư mục `btl AI` (gồm file code `train_model.py`, notebook `SeoulBikeDemand_Colab.ipynb`, web UI `index.html`, và các file mô hình đi kèm), dưới đây là tổng hợp chi tiết những gì **bạn đã sử dụng và thực hiện** tương ứng với từng yêu cầu:

---

## 🔹 Yêu cầu 1: Đọc dữ liệu, kiểm tra chất lượng, tạo đặc trưng và mã hóa

### 1. Đọc dữ liệu & Kiểm tra chất lượng:
* **Bộ dữ liệu:** `SeoulBikeData.csv` (8,760 dòng dữ liệu theo từng giờ tại Seoul, 2017–2018).
* **Đọc dữ liệu:** Dùng thư viện `pandas.read_csv()` với tùy chọn `encoding='unicode_escape'` để xử lý ký tự nhiệt độ (`°C`).
* **Chuẩn hóa tên cột:** Đổi các tên cột gốc sang tên tiếng Anh ngắn gọn: `Date`, `target`, `Hour`, `temp`, `humidity`, `wind_speed`, `visibility`, `dew_point`, `solar_rad`, `rainfall`, `snowfall`, `Seasons`, `Holiday`, `func_day`.
* **Kiểm tra chất lượng dữ liệu:**
  * Kiểm tra kích thước (`df.shape`) và giá trị thiếu qua `df.isnull().sum()` (xác nhận tập dữ liệu **không có giá trị khuyết thiếu - 0 null**).
  * Phát hiện và xử lý trường hợp ngày hệ thống không vận hành: lọc bỏ `func_day == 'No'` (295 dòng đóng cửa bảo trì dẫn đến nhu cầu bằng 0 nhân tạo), chỉ giữ lại **8,465 dòng** thực sự hoạt động (`func_day == 'Yes'`).
  * Thực hiện **5 biểu đồ EDA** chuyên sâu (lưu tại thư mục `outputs/`): phân phối biến mục tiêu & $\log(1+x)$, nhu cầu theo 24 giờ tách theo 4 mùa, phân tích theo ngày trong tuần, tương quan nhiệt độ - lượng mưa, và ma trận tương quan Heatmap.

### 2. Tạo đặc trưng (Feature Engineering):
* **Thời gian:** Chuyển `Date` sang kiểu datetime `pd.to_datetime(df['Date'], dayfirst=True)` và sắp xếp tăng dần theo thời gian (`sort_values(['Date', 'Hour'])`).
* **Đặc trưng Giờ (Hour):**
  * Giữ nguyên `Hour` (0 đến 23).
  * Áp dụng **Cyclical Encoding** (mã hóa tuần hoàn góc lượng giác) với $\sin$ và $\cos$:
    $$\text{hour\_sin} = \sin\left(\frac{2\pi \times \text{Hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \times \text{Hour}}{24}\right)$$
    giúp mô hình nắm bắt tính liền kề giữa 23:00 đêm và 00:00 sáng.
* **Đặc trưng Thứ trong tuần (Day of Week):**
  * Trích xuất `dayofweek` (0: Thứ Hai $\to$ 6: Chủ Nhật).
  * Tạo cờ nhị phân cuối tuần: `is_weekend = (dayofweek >= 5).astype(int)`.
* **Đặc trưng Tháng & Mùa:**
  * Trích xuất `month` kèm `month_sin`, `month_cos` (tuần hoàn 12 tháng).
  * Giữ nguyên cột mùa `Seasons` (Spring, Summer, Autumn, Winter).
* **Đặc trưng bổ sung:**
  * Tạo biến nhị phân thời tiết xấu: `is_rainy = (rainfall > 0)`, `is_snowy = (snowfall > 0)`.
  * Mã hóa ngày lễ: `Holiday_enc = (Holiday == 'Holiday').astype(int)`.

### 3. Mã hóa biến phân loại & Chuẩn hóa:
* Sử dụng `ColumnTransformer` của thư viện `scikit-learn`:
  * **Biến phân loại (`Seasons`):** Dùng `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` để chuyển 4 mùa thành các vector nhị phân.
  * **Biến số liên tục (`NUM_FEATURES`):** Dùng `StandardScaler()` để chuẩn hóa thang đo về phân phối chuẩn có kỳ vọng 0, độ lệch chuẩn 1.

---

## 🔹 Yêu cầu 2: Chia dữ liệu theo thời gian & Xây baseline

### 1. Chia dữ liệu theo chuỗi thời gian (Chronological Split):
* Không sử dụng phép xáo trộn ngẫu nhiên (`shuffle=False`), chia dữ liệu theo đúng dòng thời gian thực tế với tỷ lệ **60% - 20% - 20%**:
  * **Tập Train (60%):** 5,079 dòng (từ 01/12/2017 đến 02/07/2018) dùng để huấn luyện tham số mô hình.
  * **Tập Validation (20%):** 1,693 dòng (từ 02/07/2018 đến 11/09/2018) dùng để kiểm định và lựa chọn mô hình tốt nhất.
  * **Tập Test (20%):** 1,693 dòng (từ 11/09/2018 đến 30/11/2018) đóng vai trò tập kiểm thử độc lập cuối cùng.

### 2. Xây dựng mô hình Baseline:
Bạn đã xây dựng **cả 2 phương án baseline**:
1. **Baseline 1 - Trung bình nhu cầu theo giờ (Hour Mean):**
   * Tính trung bình số lượt thuê của từng khung giờ (0h đến 23h) **chỉ trên tập Train** (`train_df.groupby('Hour')['target'].mean().to_dict()`, lưu tại `baseline_hour_mean.json`).
   * Ánh xạ vào tập Validation và Test để làm mốc so sánh tối thiểu.
   * *Kết quả trên tập Test:* $\text{MAE} \approx 392.68$, $\text{RMSE} \approx 523.31$, $R^2 \approx 0.2376$.
2. **Baseline 2 - Hồi quy tuyến tính (Linear Regression):**
   * Đóng gói bộ tiền xử lý và `LinearRegression()` trong một `Pipeline`.
   * Hạn chế giá trị âm bằng hàm `np.maximum(0, y_pred)`.
   * *Kết quả trên tập Test:* $\text{MAE} \approx 302.02$, $\text{RMSE} \approx 429.63$, $R^2 \approx 0.4861$.

---

## 🔹 Yêu cầu 3: Huấn luyện mô hình cải tiến

Bạn đã huấn luyện **cả 2 mô hình cải tiến** được gợi ý trong đề bài (thay vì chỉ 1):
1. **Random Forest Regressor:**
   * Sử dụng `sklearn.ensemble.RandomForestRegressor` (`n_estimators=200`, `min_samples_leaf=2`, `random_state=42`).
   * Trích xuất biểu đồ đánh giá mức độ quan trọng đặc trưng (`feature_importance.png`), cho thấy nhiệt độ, giờ trong ngày và bức xạ mặt trời là các yếu tố chi phối mạnh nhất.
   * *Kết quả trên tập Test:* $\text{MAE} \approx 243.70$, $\text{RMSE} \approx 335.52$, $R^2 \approx 0.6866$.
2. **HistGradientBoostingRegressor:**
   * Sử dụng `sklearn.ensemble.HistGradientBoostingRegressor` (`max_iter=500`, `learning_rate=0.05`, `max_depth=6`, `min_samples_leaf=10`, `random_state=42`).
   * *Kết quả trên tập Test:* $\text{MAE} \approx 235.19$, $\text{RMSE} \approx 317.89$, $R^2 \approx 0.7187$.
* **Mô hình tối ưu nhất:** `HistGradientBoostingRegressor` đạt RMSE thấp nhất trên cả tập Validation (308.33) và Test (317.89), vượt trội so với Baseline. Mô hình này đã được bạn lưu lại thành file pipeline hoàn chỉnh `best_model_pipeline.pkl` bằng thư viện `joblib`.

---

## 🔹 Yêu cầu 4: Chống rò rỉ dữ liệu (No Data Leakage / Future Leakage)

Bạn đã tuân thủ chặt chẽ yêu cầu này thông qua các kỹ thuật:
1. **Không đưa biến đích (`target`) hay biến dẫn xuất tương lai vào tập $X$:**
   * Biến `target` được tách riêng hoàn toàn (`y_train`, `y_val`, `y_test`). Các cột đặc trưng $X$ chỉ chứa các biến ngữ cảnh thời tiết, lịch và thời gian đã biết trước hoặc tại thời điểm cần dự báo.
2. **Bảo toàn thứ tự thời gian:**
   * Dữ liệu được cắt tuần tự theo dòng thời gian (quá khứ huấn luyện cho tương lai), không dùng hàm trễ/dẫn trước (lag/lead) từ tương lai đưa về quá khứ.
3. **Tuyệt đối không lấy thống kê toàn bộ dữ liệu:**
   * Từ điển trung bình nhu cầu của Baseline (`Hour Mean`) chỉ tính thuần túy trên `df_train`.
   * Cả `StandardScaler` và `OneHotEncoder` chỉ được gọi lệnh `.fit()` trên `X_train` thông qua `Pipeline`. Dữ liệu tập Validation và Test chỉ được gọi lệnh `.transform()` theo các tham số đã học từ tập Train, tránh rò rỉ phân phối (distribution leakage).

---

## 🔹 Yêu cầu 5: Tạo giao diện / form nhập dự báo nhu cầu

Bạn đã xây dựng **2 hình thức giao diện/form** tương tác:

1. **Form tương tác trực tiếp trên Google Colab Notebook (`SeoulBikeDemand_Colab.ipynb` - Cell 36):**
   * Sử dụng hệ thống `@title` và `@param` của Colab Forms tạo giao diện UI trực tiếp:
     * **Thanh trượt (Slider):** Giờ trong ngày (`hour`: 0–23), Tháng (`month`: 1–12).
     * **Menu chọn (Dropdown):** Thứ trong tuần (`day_of_week`), Mùa (`seasons`: Xuân/Hạ/Thu/Đông), Ngày lễ (`is_holiday`: Yes/No).
     * **Ô nhập số:** Nhiệt độ (`temperature`), Độ ẩm (`humidity`), Tốc độ gió, Bức xạ mặt trời, Lượng mưa, Lượng tuyết.
   * Khi chạy ô, code tự động mã hóa đặc trưng và đưa qua `hgb_pipe.predict()` để in ra số lượt thuê xe dự kiến/giờ.

2. **Ứng dụng Web độc lập (`index.html`):**
   * Giao diện Dashboard hiện đại bằng HTML5/CSS3/JavaScript (hỗ trợ Dark Mode).
   * **Form nhập liệu:**
     * Thanh trượt kéo thả thời gian thực cho: Giờ (0–23h), Nhiệt độ (-20°C đến 40°C), Độ ẩm, Tốc độ gió, Bức xạ mặt trời, Tầm nhìn, Điểm sương, Lượng mưa, Lượng tuyết.
     * Hộp chọn: Mùa, Thứ trong tuần, Ngày lễ.
   * **Kết quả đầu ra:**
     * Hiển thị số lượt thuê dự kiến (lượt/giờ) kèm thanh đo phân loại mức độ nhu cầu (Rất thấp $\to$ Rất cao).
     * Phân tích các yếu tố ảnh hưởng cụ thể (giờ cao điểm, tác động thời tiết, cuối tuần).
     * Đưa ra khuyến nghị điều hành thực tế (bố trí thêm xe, chuẩn bị bảo dưỡng).
     * Tích hợp các tab xem kết quả đánh giá mô hình (MAE, RMSE, $R^2$ từ `model_metrics.json`) và xem lại các biểu đồ EDA.

---

## 📋 Bảng tổng hợp công nghệ & thư viện bạn đã dùng:
* **Ngôn ngữ:** Python 3, HTML5, CSS3, JavaScript.
* **Xử lý dữ liệu:** `pandas`, `numpy`.
* **Trực quan hóa:** `matplotlib`, `seaborn`.
* **Machine Learning:** `scikit-learn` (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `LinearRegression`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, `mean_absolute_error`, `mean_squared_error`, `r2_score`).
* **Triển khai & Lưu trữ:** `joblib` (lưu model pipeline), `json` (lưu metrics và baseline).
