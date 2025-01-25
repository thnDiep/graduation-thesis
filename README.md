# DỰ ĐOÁN XU HƯỚNG THỊ TRƯỜNG TÀI CHÍNH DỰA VÀO HỌC TĂNG CƯỜNG SÂU (PREDICTING FINANCIAL MARKET TRENDS BASED ON DEEP REINFORCEMENT LEARNING)
## Giới thiệu 
Dự đoán xu hướng thị trường tài chính là một nhiệm vụ phức tạp và đầy thách thức, ngay cả đối với các nhà đầu tư có kinh nghiệm. Trong đó, phân tích kỹ thuật là một trong những phương pháp phổ biến được sử dụng để dự đoán giá và xu hướng cổ phiếu. Gần đây, các mô hình học máy, đặc biệt là học tăng cường sâu (Deep reinforcement learning - DRL), đã nhận được sự quan tâm đáng kể trong lĩnh vực này. Các nghiên cứu trước đây đã tích hợp dữ liệu phân tích kỹ thuật với thông tin lịch sử giá vào các mô hình học tăng cường nhưng vẫn chưa đạt được kết quả nổi bật. Trong nghiên cứu này, nhóm em đã áp dụng một cách tiếp cận mới bằng cách phân loại các chỉ báo kỹ thuật thành hai loại: chỉ báo sớm và chỉ báo trễ. Mỗi loại chỉ báo được xử lý qua một mô hình học có giám sát để tạo ra tín hiệu BUY/NONE/SELL, kết hợp với thông tin lịch sử giá làm đầu vào cho mô hình học tăng cường sâu. Bên cạnh đó, nhóm còn nghiên cứu thêm tác động của việc sử dụng thông tin từ các mức Hỗ trợ và Kháng cự trong học tăng cường sâu, áp dụng trên cùng một khung thời gian và trên nhiều khung thời gian khác nhau. Kết quả thử nghiệm cho thấy việc tích hợp hai nguồn tín hiệu dự đoán bởi mô hình CNN-2D vào mô hình học tăng cường sâu mang lại lợi nhuận cao hơn so với mô hình cơ sở và các mô hình học máy phổ biến khác. Hiệu suất của mô hình vượt trội hơn nhờ việc kết hợp một cách hiệu quả các tín hiệu từ các chỉ báo kỹ thuật. Đồng thời, nghiên cứu cũng chỉ ra rằng thông tin từ các mức Hỗ trợ và Kháng cự có tác động tích cực lên hiệu suất giao dịch của mô hình.

## Cấu trúc dự án
Thư mục `SOURCE` chứa toàn bộ các tập tin và tài nguyên liên quan để có thực thi mã nguồn, bao gồm:
* Thư mục `BaselineModels` và Thư mục `BaselineSequentialModels`: chứa các mô hình cơ sở lấy từ nghiên cứu của [Taghian và cộng sự](https://github.com/MehranTaghian/DQN-Trading), được dùng làm mô hình gốc để phát triển thêm và để so sánh với mô hình và các phương pháp đề xuất.
* Thư mục `Data`: bao gồm 6 tập dữ liệu là AAL, BTC-USD, GE, GOOGL, SH1A0001, và SZ399005.
* Thư mục `DataLoader`: thực hiện truy xuất dữ liệu và tiền xử lý dữ liệu
* Thư mục `Models`: lưu các mô hình đã được huấn luyện.
* Thư mục `Results`: chứa các tập tin kết quả thực nghiệm.
  * Các thư mục là tên thị trường: bao gồm kết quả giao dịch của các mô hình trên thị trường tương ứng.
  * Thư mục `actions`: lưu các hành động mô hình đã thực hiện trên tập huấn luyện và kiểm tra.
* `Action.py`: định nghĩa lớp enum "Action" đại diện cho 3 hành động MUA/BÁN/KHÔNG.
* `CNNModel.py`: xây dựng kiến trúc mô hình CNN-2D dựa trên nghiên cứu của [Hoseinzade và Haratizadeh](https://www.sciencedirect.com/science/article/abs/pii/S0957417419301915).
* `Evaluation.py`: tính toán các độ đo đánh giá hiệu quả giao dịch thông qua các hành động được dự đoán từ các mô hình.
* `main.py`: điểm vào của ứng dụng, sử dụng các biến được gán cố định và các tham số đầu vào thực hiện huấn luyện hoặc kiểm tra mô hình thông qua lớp `SensitivityRun`.
* `ReplayMemory.py`: định nghĩa lớp ReplayMemory, sử dụng trong học tăng cường sâu, để lưu trữ và trích xuất các trải nghiệm.
* `resample.py`: chạy riêng tập tin này để downsampling dữ liệu gốc (`SH1A0001_1m.csv` và `SZ399005_1m.csv` được lấy từ nguồn của [Liu và cộng sự](https://github.com/marscrazy/MTDNN/tree/master)) từ khung thời gian 1 phút thành các khung thời gian 5 phút, 30 phút, và 2 giờ rồi lưu vào các tập tin khác nhau để sử dụng sau này.
* `SensitivityRun.py`: định nghĩa lớp `SensitivityRun`, có nhiệm vụ khởi tạo dữ liệu, khởi tạo mô hình, huấn luyện và kiểm thử mô hình.
* `utils.py`: định nghĩa các hàm hỗ trợ.


## Yêu cầu hệ thống
* Python phiên bản 3.11
* Cài đặt các thư viện cần thiết trong `requirements.txt` (tập tin này nằm trong thư mục `LIB`).

## Hướng dẫn chạy mã nguồn
Để chạy mã nguồn trong thư mục `SOURCE`, sử dụng câu lệnh:
```shell
python main.py -t <trader> -m <model> -w <window_size> -d <dataset> -n <nep>
```
* `trader` bao gồm các loại đào tạo mô hình. Mặc định là `test`.
  * `train`: Chỉ huấn luyện mô hình. Sử dụng tùy chọn này khi bạn muốn mô hình học từ dữ liệu huấn luyện và cập nhật các tham số của nó.
  * `test`: Chỉ kiểm tra mô hình. Sử dụng tùy chọn này khi bạn đã huấn luyện mô hình và muốn đánh giá hiệu suất của nó.
  * `train_test`: Huấn luyện và kiểm tra mô hình. Sử dụng tùy chọn này khi bạn muốn thực hiện cả hai bước: huấn luyện mô hình trên tập dữ liệu huấn luyện và sau đó đánh giá nó trên tập dữ liệu kiểm tra.
* `model` đại diện cho tên mô hình. Mặc định là `DQN_TI_SIGNAL`.
  * `2D-CNN_PI`, `2D-CNN_CI`: hai mô hình CNN-2D sử dụng đầu vào là chỉ báo sớm và chỉ báo trễ.
  * `DQN`: mô hình cơ sở sử dụng thông tin lịch sử giá (OHLC).
  * `DQN_PI`, `DQN_CI`, `DQN_TI`: mô hình DRL kết hợp thêm thông tin các loại chỉ báo kĩ thuật (chỉ báo sớm, chỉ báo trễ, cả chỉ báo sớm và trễ).
  * `DQN_SRP_COL`, `DQN_SRS_COL`: mô hình DRL kết hợp thêm thông tin từ các mức Hỗ trợ và Kháng cự, mở rộng trạng thái môi trường theo chiều đặc trưng.
  * `DQN_SRP_ROW`, `DQN_SRS_ROW`: mô hình DRL kết hợp thêm thông tin từ các mức Hỗ trợ và Kháng cự, mở rộng trạng thái môi trường theo chuỗi thời gian.
  * `DQN_MTF_SRP`, `DQN_MTF_SRS`: mô hình DRL kết hợp thêm thông tin từ các mức Hỗ trợ và Kháng cự trên nhiều khung thời gian (5 phút, 30 phút, 2 giờ), mở rộng trạng thái môi trường theo chuỗi thời gian.
  * `DQN_TI_SIGNAL`: mô hình được đề xuất trong nghiên cứu (Ours).
  * `RF`, `SVM`, `HARD_VOTING`: các mô hình phổ biến được thử nghiệm để so sánh.
  * `DQN_CNN`: mô hình cơ sở sử dụng bộ trích xuất đặc trưng CNN, sử dụng thông tin lịch sử giá (OHLC).
  * `DQN_CNN_SRP_COL`, `DQN_CNN_SRS_COL`: mô hình DRL có bộ trích xuất đặc trưng CNN, kết hợp thêm thông tin từ các mức Hỗ trợ và Kháng cự, mở rộng trạng thái môi trường theo chiều đặc trưng.
  * `DQN_CNN_SRP_ROW`, `DQN_CNN_SRS_ROW`: mô hình DRL có bộ trích xuất đặc trưng CNN, kết hợp thêm thông tin từ các mức Hỗ trợ và Kháng cự, mở rộng trạng thái môi trường theo chuỗi thời gian.
* `window_size` quy định kích thước cửa sổ. Mặc định là `10`.
* `dataset` bao gồm 6 tập dữ liệu `AAL`, `BTC-USD`, `GE`, `GOOGL`, `SH1A0001`, và `SZ399005`. Mặc định là `BTC-USD`.
* `nep` viết tắt của number of episodes là số lượng tập huấn luyện của mô hình. Mặc định là `1`.

Ví dụ câu lệnh cần chạy để kiểm tra mô hình đề xuất trên tập dữ liệu AAL:
```shell
python main.py -m DQN_TI_SIGNAL -d AAL
```
Sau khi chạy câu lệnh trên, tập tin kết quả `DQN_TI_SIGNAL-indicator.txt` sẽ được tạo ra trong thư mục `Results/AAL`, chứa kết quả hiệu suất giao dịch của quá trình kiểm tra mô hình.
## Trích dẫn
```bibtex
@inproceedings{tran2024enhancing,
  title={Enhancing Financial Market Prediction with Reinforcement Learning and Ensemble Learning},
  author={Tran, Diep and Tran, Quyen and Tran, Quy and Nguyen, Vu and Tran, Minh-Triet},
  booktitle={IFIP International Conference on Artificial Intelligence Applications and Innovations},
  pages={32--46},
  year={2024},
  organization={Springer}
}
```
