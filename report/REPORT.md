# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/9/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Ultralytics - 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
    "class_id": 468,
    "class_name": "cab",
    "rank": 1,
    "score": 0.510915
    "taxonomy_name": "ImageNet-1K",
- Record này mô tả toàn ảnh như thế nào?
    cab
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    yolo11n-cls.pt
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    class_id: định danh bằng ID, thuận tiện cho máy xử lý.
    class_name: người đọc biết vật thể detect ra  là thuộc loại gì.
    taxonomy_name: cho biết vật thể nhận thuộc hệ thống phân loại nào.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    guideline cần quy định cách chọn nhãn cho toàn ảnh khi có nhiều chủ thể. Ví dụ như nếu có nhiều chủ thể thì làm như nào, một vật có thể có bao nhiêu nhãn
- Vì sao model score không phải ground truth?
    Vì model score chỉ là độ tin cậy mà model dự đoán vật sẽ thuộc loại label nào.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    "class_name": "bus",
    "score": 0.912558,
    "bbox_xyxy": [
      93.17,
      187.95,
      223.01,
      320.91
    ],
    "bbox_width": 129.84,
    "bbox_height": 132.96
- Diễn giải vị trí box bằng lời:
    Box bắt đầu từ pixel có tọa độ (93.17, 187.95) đến pixel có tọa độ (223.01, 320.91)
- So sánh số prediction ở hai threshold:
    với ảnh traffic, nếu threshold = 0.35 cho ra 26 vật, nếu threshold = 0.5 cho ra 18 vật
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    Threshold thấp cho độ bao phủ lớn, lấy được nhiều vật thể, khối lượng reviewer cần xem nhiều hơn.
    Threshold cao cho độ bao phủ nhỏ, lấy được ít vật thể, khối lượng reviewer cần xem ít hơn.
- Đề xuất một quy tắc box chặt:
    Box phải chứa toàn bộ vật thể, khoảng cách cạnh đến điểm gần nhất của vật phải là nhỏ nhất
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    Guideline
        Quy định khung của object có giới hạn đến đâu
        Quy định ngưỡng tối thiểu cần nhìn thấy của vật thể để cho vào detection
        Quy định việc detect một vật bị che, làm tách thành 2 nửa  
    Escalation khi:
        Cắt mép ảnh quá sâu: Đối tượng nằm sát biên ảnh chỉ lộ một chi tiết không đủ định danh
        Hình ảnh phản chiếu hoặc tranh vẽ (Ghost objects): Tranh quảng cáo trên thân xe buýt có hình người, hoặc hình ảnh phản chiếu qua gương/cửa kính ô tô.
        Cụm đối tượng chồng lấn phức tạp

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    "instance_id": "traffic-001",
    "class_name": "bus",
    "score": 0.925745,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],
      [
        142.0,
        192.0
      ],
      [
        141.0,
        193.0
      ],
      [
        139.0,
        193.0
      ],
      [
        137.0,
        195.0
      ],
      [
        136.0,
        195.0
      ],
      [
        135.0,
        196.0
      ],
      [
        134.0,
        196.0
      ],
      [
        133.0,
        197.0
      ],
      [
        132.0,
        197.0
      ],
      [
        131.0,
        198.0
      ],
      [
        130.0,
        198.0
      ],
      [
        128.0,
        200.0
      ],
      [
        127.0,
        200.0
      ],
      [
        126.0,
        201.0
      ],
      [
        125.0,
        201.0
      ],
      [
        123.0,
        203.0
      ],
      [
        122.0,
        203.0
      ],
      [
        121.0,
        204.0
      ],
      [
        120.0,
        204.0
      ],
      [
        119.0,
        205.0
      ],
      [
        117.0,
        205.0
      ],
      [
        116.0,
        206.0
      ],
      [
        115.0,
        206.0
      ],
      [
        114.0,
        207.0
      ],
      [
        113.0,
        207.0
      ],
      [
        111.0,
        209.0
      ],
      [
        110.0,
        209.0
      ],
      [
        109.0,
        210.0
      ],
      [
        108.0,
        210.0
      ],
      [
        105.0,
        213.0
      ],
      [
        104.0,
        213.0
      ],
      [
        98.0,
        219.0
      ],
      [
        98.0,
        220.0
      ],
      [
        96.0,
        222.0
      ],
      [
        96.0,
        312.0
      ],
      [
        97.0,
        313.0
      ],
      [
        97.0,
        314.0
      ],
      [
        98.0,
        315.0
      ],
      [
        101.0,
        315.0
      ],
      [
        102.0,
        316.0
      ],
      [
        104.0,
        316.0
      ],
      [
        105.0,
        317.0
      ],
      [
        106.0,
        317.0
      ],
      [
        107.0,
        318.0
      ],
      [
        108.0,
        318.0
      ],
      [
        109.0,
        319.0
      ],
      [
        113.0,
        319.0
      ],
      [
        114.0,
        318.0
      ],
      [
        116.0,
        318.0
      ],
      [
        117.0,
        317.0
      ],
      [
        122.0,
        317.0
      ],
      [
        123.0,
        316.0
      ],
      [
        131.0,
        316.0
      ],
      [
        132.0,
        315.0
      ],
      [
        163.0,
        315.0
      ],
      [
        164.0,
        314.0
      ],
      [
        170.0,
        314.0
      ],
      [
        171.0,
        315.0
      ],
      [
        176.0,
        315.0
      ],
      [
        177.0,
        316.0
      ],
      [
        183.0,
        316.0
      ],
      [
        184.0,
        315.0
      ],
      [
        185.0,
        315.0
      ],
      [
        188.0,
        312.0
      ],
      [
        188.0,
        311.0
      ],
      [
        190.0,
        309.0
      ],
      [
        190.0,
        308.0
      ],
      [
        197.0,
        301.0
      ],
      [
        198.0,
        301.0
      ],
      [
        199.0,
        300.0
      ],
      [
        201.0,
        300.0
      ],
      [
        202.0,
        299.0
      ],
      [
        203.0,
        299.0
      ],
      [
        204.0,
        298.0
      ],
      [
        205.0,
        298.0
      ],
      [
        206.0,
        297.0
      ],
      [
        209.0,
        297.0
      ],
      [
        210.0,
        296.0
      ],
      [
        211.0,
        296.0
      ],
      [
        213.0,
        294.0
      ],
      [
        213.0,
        293.0
      ],
      [
        214.0,
        292.0
      ],
      [
        214.0,
        291.0
      ],
      [
        215.0,
        290.0
      ],
      [
        215.0,
        289.0
      ],
      [
        217.0,
        287.0
      ],
      [
        217.0,
        286.0
      ],
      [
        219.0,
        284.0
      ],
      [
        219.0,
        283.0
      ],
      [
        220.0,
        282.0
      ],
      [
        220.0,
        281.0
      ],
      [
        221.0,
        280.0
      ],
      [
        221.0,
        279.0
      ],
      [
        222.0,
        278.0
      ],
      [
        222.0,
        276.0
      ],
      [
        223.0,
        275.0
      ],
      [
        223.0,
        234.0
      ],
      [
        222.0,
        233.0
      ],
      [
        222.0,
        226.0
      ],
      [
        221.0,
        225.0
      ],
      [
        221.0,
        220.0
      ],
      [
        220.0,
        219.0
      ],
      [
        220.0,
        208.0
      ],
      [
        219.0,
        207.0
      ],
      [
        219.0,
        202.0
      ],
      [
        218.0,
        201.0
      ],
      [
        218.0,
        198.0
      ],
      [
        217.0,
        197.0
      ],
      [
        217.0,
        196.0
      ],
      [
        213.0,
        192.0
      ],
      [
        211.0,
        192.0
      ],
      [
        210.0,
        191.0
      ],
      [
        206.0,
        191.0
      ],
      [
        205.0,
        190.0
      ],
      [
        199.0,
        190.0
      ],
      [
        198.0,
        189.0
      ]
    ]

- Polygon bổ sung chi tiết gì so với box?
    Polygon bổ sung:         
        Biên dạng hình học chính xác, mô tả chính xác đường viền thực tế theo hình dạng tự nhiên của đối tượng.
        Loại bỏ nhiễu nền, ôm sát mép vật thể, tách biệt chủ thể khỏi hậu cảnh.
        Tư thế và góc nghiêng, phản ánh trực tiếp hướng xoay, độ nghiêng và tư thế không gian của vật thể.
        Khả năng biểu diễn cấu trúc không liền khối như khi vật thể bị chia cắt làm nhiều phần nhìn thấy, polygon có thể tách thành các đa giác con trên cùng một thực thể, tránh việc phải bao trùm cả vật chắn như box. 
- `instance_id` dùng để làm gì và không phải loại ID nào?
    instance_id dùng để định danh từng vật thể trong một ảnh hoặc xuyên suốt các frame video. Phân biệt giữa các vật thể thuộc cùng một lớp (ví dụ: cùng là car). Gom nhóm các phần đứt rời của cùng một vật thể bị che khuất thành một thực thể duy nhất.
    instance_id không phải 
        category_id / class_id
        image_id / sample_id
        annotation_id
- Đề xuất một quy tắc biên mask:
    Quy tắc 50% Pixel Cutoff: Một pixel nằm trên đường viền chỉ được tính vào mask nếu diện tích thuộc về vật thể chiếm từ 50% trở lên của pixel đó.
    Tối thiểu hóa điểm gãy: Tăng mật độ điểm ở các đoạn cua gắt/chi tiết nhỏ (2–3 px/điểm) và giảm mật độ ở các cạnh thẳng dài.
    Loại trừ khoảng rỗng nội tại: Các khoảng trống bên trong thân đối tượng phải được đục lỗ, không bôi kín toàn bộ bề mặt.
    Dung sai kiểm định biên: Độ lệch của đường biên mask không được vượt quá một lượng pixel đã quy định.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    Vùng mờ chuyển động hoặc mất nét cần quy định chọn biên tại đường trung tâm chuyển tiếp của vệt mờ hay bao trùm toàn bộ dải mờ tỏa ra ngoài.
    Vật thể tiếp xúc/dính sát nhau không được để chồng lấn giữa 2 vùng bao phủ khác nhau trên cùng một pixel. Cần quy định rõ bề dày đường phân giới giữa 2 vật thể sát nhau.
    Vùng che khuất chỉ gắn mask cho phần thực tế nhìn thấy hay suy đoán cả phần bị che. Nếu một vật thể bị chia cắt thành 2 mảng nhỏ, mảng nhỏ có diện tích dưới ngưỡng bao nhiêu % thì được phép bỏ qua không cần vẽ.
    Các trường hợp cần Escalation
        Độ tương phản quá thấp 
        Vật thể bán trong suốt như kính xe, rèm voan, hoặc chất lỏng trong cốc — cần quyết định mask tính cho lớp kính hay đối tượng phía sau kính.
        Cấu trúc dạng sợi/mảnh như tóc bay, lông động vật, cành lá rậm rạp — cần quyết định quy chuẩn vẽ gộp hay vẽ chi tiết từng sợi/nhánh.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| ---   | --- | --- | --- | --- |
| Phân loại ảnh | - Chuỗi nhãn / Category ID đơn lẻ hoặc đa nhãn (Multi-label). - Định dạng: CSV, JSON ({"image_id": ..., "labels": [...]}), hoặc cấu trúc thư mục dạng tên class (dataset/train/<class_name>/). | - Ảnh chứa nhiều đối tượng thuộc các nhãn khác nhau nhưng chỉ được gán single-label (không rõ chủ thể chính). - Đối tượng chính quá nhỏ, nằm ngoài rìa hoặc bị chìm vào nền. - Các lớp dễ gây nhầm lẫn ngữ nghĩa (ví dụ: sedan vs. hatchback, báo đốm vs. báo hoa mai). | - Nhận diện đối tượng trọng tâm chiếm ưu thế trong ảnh. - Chọn 1 hoặc nhiều nhãn phù hợp từ danh mục taxonomy có sẵn. - Gắn cờ (flag/escalate) nếu ảnh bị hỏng, quá tối, hoặc không thể xác định chủ thể. | - Tính chính xác của nhãn so với taxonomy và guideline ngữ nghĩa. - Tính nhất quán giữa các annotator đối với các ca biên (edge cases). - Kiểm tra xem annotator có nhầm lẫn giữa nhãn chủ thể chính và vật thể phụ ở hậu cảnh hay không. |
| Phát hiện vật thể | - Tọa độ Bounding Box 2D dạng hình chữ nhật trục thẳng đứng. - Định dạng phổ biến: COCO ([x_min, y_min, width, height]), YOLO ([class_id, x_center, y_center, w, h] chuẩn hóa 0–1), hoặc Pascal VOC ([xmin, ymin, xmax, ymax]). | - Box quá rộng làm thừa nền hoặc box quá chật cắt lẹm vào chi tiết vật thể. - Đóng khung cả bóng đổ hoặc hình ảnh phản chiếu qua gương. - Bỏ sót đối tượng nhỏ/ở xa hoặc vẽ trùng nhiều box đè lên cùng 1 đối tượng. - Không rõ quy tắc vẽ khi vật thể bị che khuất chia làm 2 khúc (vẽ 1 box to hay 2 box con). | - Kéo thả khung chữ nhật bao trùm toàn bộ pixel nhìn thấy được của từng thực thể. - Gán đúng nhãn (class_id) cho từng box. - Xóa bỏ box thừa, tách các box bị dính chùm. - Tuân thủ quy tắc đóng khung vùng thực tế nhìn thấy (visible) hay ước lượng toàn phần (amodal). | - Độ khít của box,IoU với ground truth chuẩn, sai số pixel biên (1–2 px).- Độ bao phủ (Coverage/Recall): kiểm tra xem có vật thể nào bị annotator bỏ sót không.- Kiểm tra lỗi dương tính giả (vẽ box vào bóng râm, tranh ảnh trên tường, vệt sáng).- Đúng nhãn phân loại của từng box. |
| Instance segmentation | - Tập hợp tọa độ đỉnh đa giác (Polygon vertices: [[x1, y1, x2, y2, ...]]) hoặc mặt nạ nhị phân mã hóa RLE (Run-Length Encoding). - Kèm theo instance_id duy nhất cho từng cá thể. | - Đường biên răng cưa, thiếu điểm neo ở các đoạn cong hoặc thừa điểm không cần thiết trên cạnh thẳng. - Bỏ qua các khoảng rỗng nội tại (không đục lỗ khoảng trống giữa tay và thân người, nan hoa xe). - Hai mask của 2 đối tượng đứng cạnh nhau bị chồng lấn pixel (overlap) hoặc để hở rãnh nền. - Biên mờ/nhòe (motion blur) hoặc cấu trúc dạng sợi (tóc, cành lá) khó chốt ranh giới. | - Chấm các điểm đỉnh tạo polygon ôm sát ranh giới pixel của từng đối tượng riêng biệt. - Vẽ các vòng đa giác trong (interior rings) để đục lỗ các khoảng rỗng bên trong. - Gán instance_id riêng biệt cho từng đối tượng; gán chung một instance_id cho các mảng rời rạc thuộc cùng 1 vật thể bị vật khác chắn ngang. - Đảm bảo không để pixel chồng lấn sang cá thể bên cạnh. | - Độ chuẩn xác của biên mask (Boundary IoU / Boundary F1-score), kiểm tra dưới độ zoom cao (300%).- Độ sạch của các khoảng rỗng (đã đục lỗ đầy đủ chưa).- Tính duy nhất và liên kết của instance_id (đặc biệt ở các vật thể bị chia cắt thành nhiều phần).- Phân xử quyền sở hữu pixel tại các ranh giới tiếp xúc sát nhau (không bị overlap). |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
    Tuân thủ nguyên tắc bảo mật và quyền riêng tư: Tuyệt đối không sao chép, tải về thiết bị cá nhân, chụp màn hình hoặc chia sẻ dữ liệu/ảnh dự án ra bên ngoài môi trường làm việc được cấp phép; làm mờ hoặc ẩn các thông tin định danh cá nhân nếu dữ liệu chưa được ẩn danh hóa.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
    Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho cấp trên (TechLead, PM,...)

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
