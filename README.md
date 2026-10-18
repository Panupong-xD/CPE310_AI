# GlycoRisk — รายงานตรวจสอบและคู่มือโปรเจค Data Science ตั้งแต่ข้อมูลดิบจนถึง Web Application

เอกสารฉบับนี้อธิบายโค้ดที่แก้ไขและรันจริงเมื่อ **18 ตุลาคม 2026** ตามเวลาประเทศไทย
อ่านได้โดยไม่ต้องเคยทำโปรเจคนี้มาก่อน
มีทั้งเหตุผลทาง Data Science โค้ด สูตร ผลของทุกโมเดล และวิธีตรวจสอบตัวเลขย้อนกลับ
ตัวเลขในรายงานสร้างจาก artifacts จริง ไม่ใช่ตัวเลขตัวอย่างที่เขียนขึ้นเพื่อแสดงหน้าตา

โปรเจคนี้เป็น **โมเดลเพื่อการศึกษา สำหรับจำแนก diabetes proxy ณ ช่วงเวลาที่สำรวจ**
ใช้ข้อมูล NHANES สหรัฐอเมริกา รอบ 2013–2014
คำว่า proxy หมายถึงสถานะที่นิยามจาก HbA1c ประวัติการวินิจฉัย และหลักฐานการรักษา
ไม่ได้มีข้อมูลติดตามว่าผู้เข้าร่วมจะเป็นเบาหวานในอนาคตเมื่อใด
จึงไม่ควรเรียกคะแนนว่าโอกาสเกิดเบาหวานใน 5 หรือ 10 ปี
ไม่ได้เป็นโมเดลวินิจฉัย prediabetes และยังไม่มี external validation ในประเทศไทย

## สารบัญ

1. [ผลตรวจภาพรวม](#ผลตรวจภาพรวม)
2. [ไฟล์และโครงสร้าง](#ไฟล์และโครงสร้าง)
3. [นิยามโจทย์](#นิยามโจทย์)
4. [ข้อมูลดิบและการตรวจแหล่งที่มา](#ข้อมูลดิบและการตรวจแหล่งที่มา)
5. [การรวมตารางและคัดเลือกผู้เข้าร่วม](#การรวมตารางและคัดเลือกผู้เข้าร่วม)
6. [นิยาม label ที่แก้ไข](#นิยาม-label-ที่แก้ไข)
7. [Cleaning และ feature engineering](#cleaning-และ-feature-engineering)
8. [พจนานุกรม features ทั้งหมด](#พจนานุกรม-features-ทั้งหมด)
9. [การแบ่งข้อมูลและป้องกัน leakage](#การแบ่งข้อมูลและป้องกัน-leakage)
10. [EDA บนชุดฝึก](#eda-บนชุดฝึก)
11. [โมเดลและ cross-validation](#โมเดลและ-cross-validation)
12. [Calibration และ threshold](#calibration-และ-threshold)
13. [ผลทดสอบของทุกโมเดล](#ผลทดสอบของทุกโมเดล)
14. [ความไม่แน่นอนและผลแยกกลุ่ม](#ความไม่แน่นอนและผลแยกกลุ่ม)
15. [SHAP และความหมายของคะแนน](#shap-และความหมายของคะแนน)
16. [Backend และการทำนาย](#backend-และการทำนาย)
17. [What-if และ frontend](#what-if-และ-frontend)
18. [วิธีรันตั้งแต่ต้น](#วิธีรันตั้งแต่ต้น)
19. [การทดสอบและตรวจซ้ำ](#การทดสอบและตรวจซ้ำ)
20. [ข้อจำกัดและแนวทางพัฒนา](#ข้อจำกัดและแนวทางพัฒนา)
21. [โค้ดฉบับเต็มและคำอธิบายรายไฟล์](#โค้ดฉบับเต็มและคำอธิบายรายไฟล์)

## ผลตรวจภาพรวม

โครงสร้างเดิมมีส่วนที่ทำถูกแล้ว เช่น ใช้ SEQN เชื่อมข้อมูลผู้เข้าร่วม
มี adult cohort มี stratified holdout และเลือกโมเดลจาก CV ROC-AUC
โมเดลที่ export เดิมเป็น Pipeline ซึ่ง fit ตัว preprocessing บน train ก่อนทำนาย test
ตัวเลข test ของ Random Forest เดิมตรวจจาก artifact ได้ตรงกับ metadata
แต่กระบวนการ CV นิยาม label การสื่อสารผล และการจำลองมีข้อผิดพลาดที่ต้องแก้

### ตรวจผลเดิมได้อะไรบ้าง

โหลดโมเดลเดิมกับ split เดิมก่อนแก้ pipeline แล้วได้ผลดังนี้:

- จำนวนผู้เข้าร่วมเดิม 5,654 คน
- Positive เดิม 1,095 คน
- Test เดิม 1,131 คน
- Random Forest test ROC-AUC = 0.7789969358327324
- Random Forest test Average Precision = 0.4432344009372194
- Confusion matrix = TN 627, FP 285, FN 62, TP 157

จึงไม่ได้สรุปว่าผลเดิมทั้งหมดเป็นตัวเลขปลอม
ประเด็นคือผลดังกล่าวประเมิน label และ preprocessing protocol เดิม
ซึ่งมีข้อจำกัดหลายรายการด้านล่าง
หลักฐานเก็บใน `audit/original_verification.json`

### ข้อผิดพลาดที่พบและสิ่งที่แก้

| ประเด็น | เดิม | แก้แล้ว |
| --- | --- | --- |
| CV preprocessing leakage | fit_transform(X_train) หนึ่งครั้งก่อนแบ่ง fold | สร้าง Pipeline ใหม่ภายในแต่ละ fold; imputer/scaler เรียนจาก fold-training เท่านั้น |
| ใช้ glucose ผิดชนิด | เรียก LBXSGL ว่า fasting glucose และตัดที่ 126 | เอา LBXSGL ออกจากกฎ label; ใช้ HbA1c/diagnosis/therapy proxy ที่มีหลักฐานในไฟล์ |
| control ไม่มี biomarker | ยอมให้ HbA1c และ glucose หายทั้งคู่แล้วยังเป็น negative | negative ต้องมี HbA1c <6.5 ที่สังเกตจริง และ DIQ010=2 |
| ค้นหายากว้างเกินไป | ค้น GLUCO/DIABET ในชื่อยาและเหตุผลที่ใช้ยา | จับชื่อสารออกฤทธิ์เต็มคำใน RXDDRUG และ RXDUSE=1; aggregate เป็นรายคน |
| ตาราง benchmark เดิมปะปน | recall/AP บางรายการเป็นค่า CV แต่แสดงเหมือน test | export ผล test ของทุก candidate แยกจากผล CV ชัดเจน |
| ไม่มี baseline ที่ไม่ใช้ features | เปรียบเทียบเฉพาะโมเดล ML | เพิ่ม DummyPrior เพื่อดูค่า AUC/AP/Brier ที่ต้องชนะ |
| class weighting กับ probability | นำ predict_proba ของ balanced classifier แสดงเป็นความเสี่ยงตรง ๆ | ใช้ sigmoid calibration บน partition แยกจาก model fitting |
| threshold 0.5 โดยไม่มีเหตุผล | ใช้ 0.5 กับทุกโมเดลทั้งที่สเกล probability ต่างกัน | เลือก threshold จาก OOF ของชุดฝึก ให้ sensitivity อย่างน้อย 80% และ specificity สูงสุดภายใต้กฎนี้ |
| statistics ใช้ทั้ง cohort | ค่า median fallback รวม test เข้าไปด้วย | feature_stats สำหรับ serving คำนวณจาก fit partition เท่านั้น |
| เดารอบเอวเอง | ใช้ส่วนสูงคูณ 0.48 หรือ 0.54 เมื่อไม่มีรอบเอว | ส่ง missing เข้า fitted imputer; ไม่สร้างรอบเอวเสมือนว่าเป็นค่าที่วัดจริง |
| calories ที่ซ่อนใน form | state ส่งค่า 2,100 kcal ทั้งที่ไม่มีช่องให้ผู้ใช้แก้ | เพิ่ม optional calories input และให้ค่าเริ่มต้นว่าง |
| train/serve feature ต่างกัน | train ใช้ BMI ที่ปัดเศษในไฟล์ แต่ API คำนวณอีกสูตร | เรียก engineer_features ฟังก์ชันเดียวกันทั้ง training และ API |
| sugar/fiber ระเบิดเมื่อ fiber=0 | หารด้วย fiber+1e-5 ทำให้บางค่าเกิน 60 ล้าน | fiber<=0 ทำให้ ratio เป็น missing แล้วให้ fold-specific imputer จัดการ |
| activity ผิดความหมาย | PAQ605 เป็นกิจกรรมหนักจากงาน แต่ UI บอกออกกำลังกาย | ใช้ PAQ650 กิจกรรมหนักเพื่อสันทนาการ; เปลี่ยนข้อความ UI ให้ตรงคำถาม |
| ไม่ทำกิจกรรมหนัก = sedentary | ตั้งชื่อ sedentary_flag จากคำถามกิจกรรมหนักข้อเดียว | ใช้ชื่อ no_vigorous_recreation_flag; ไม่อ้างว่าเป็นการนั่งมากหรือไม่มีกิจกรรมทุกชนิด |
| unknown history กลายเป็น false | BPQ020=7/9 ถูกมองคล้ายไม่มีประวัติ | map เฉพาะ 1/2; unknown คงเป็น missing เว้นแต่ BP ให้ positive evidence |
| ปลอม protective factor | หากไม่มี SHAP ติดลบ ใส่ Dietary Fiber -3.5% เอง | คืนรายการว่างได้; ทุก factor ต้องมาจาก SHAP จริง |
| SHAP ใช้หน่วยผิด | เอาสัดส่วน abs(SHAP) ไปใส่เครื่องหมาย % | อธิบาย calibrated probability โดยตรงและใช้ percentage points หรือ pp |
| ความเข้ากันได้ของ explainer | เดิมรองรับเฉพาะต้นไม้; เมื่อเพิ่ม calibration ต้องอธิบาย wrapper ให้ครบ | ใช้ PermutationExplainer รองรับ champion ทุกชนิดและ calibrated predictor ทั้งชุด |
| pre-diabetes จาก score band | เรียกคะแนน 20–50% ว่า prediabetes alert | ใช้ชื่อช่วงคะแนนทั่วไป; แยกสีจาก threshold คัดกรอง |
| what-if อ้างเหตุและผล | ข้อความสัญญาว่าลดน้ำหนักแล้วจะลดความเสี่ยงจริงเท่านั้นเท่านี้ | รายงานการเปลี่ยน score ภายใต้ input scenario; ไม่สรุป treatment effect |
| ลบประวัติความดันใน simulation | จัดการความดันแล้วตั้ง hypertension_history=False | เปลี่ยนเฉพาะค่าความดันปัจจุบัน; ประวัติการวินิจฉัยยังอยู่ |
| reset เปลี่ยน activity โดยไม่ตั้งใจ | reset ตั้ง exercise=False แม้ baseline เป็น True | คืน activity ตาม baseline และเริ่ม sliders ที่ศูนย์ |
| claim ระดับประชากร | เรียก unweighted training percentile ว่า percentile ประชากร | ใช้ OOF reference ของชุดฝึกและระบุว่า unweighted sample |
| ความไม่แน่นอนไม่ชัด | มีเพียง point estimates | เพิ่ม stratified IID bootstrap 1,000 ครั้ง พร้อมข้อจำกัดเรื่อง survey design |

การแก้ครั้งนี้ไม่ได้ปรับสูตรหรือเลือกโมเดลเพื่อดัน test AUC ให้สูงที่สุด
โมเดลชนะถูกเลือกด้วย CV ของชุด fit ก่อนเปิดใช้ test ใน workflow ใหม่
แม้ GradientBoosting จะมี test AUC สูงกว่าเล็กน้อย ก็ไม่เปลี่ยน champion ตาม test
อย่าเปรียบเทียบ AUC เดิม 0.7790 กับ AUC ใหม่ 0.8161 แล้วสรุปว่าดีขึ้นเป็นเหตุจากการแก้เพียงอย่างเดียว
เพราะ label, cohort, partition และรายละเอียด features เปลี่ยนหลายส่วนพร้อมกัน

## ไฟล์และโครงสร้าง

```text
CPE310 AI/
├── Dataset/                         # ข้อมูล CSV ดิบ 6 ตาราง: ไม่แก้ไข
├── ml_pipeline/
│   ├── preprocess.py                # join, cohort, label, cleaning, features, preprocessor
│   ├── train.py                     # split 60/20/20, fit-only 5-fold CV, เลือก champion
│   ├── evaluate.py                  # calibration, thresholds, test metrics, bootstrap, export
│   ├── explain.py                   # callable สำหรับ SHAP ของ calibrated probability
│   ├── eda.py                       # สถิติของ fit partition
│   ├── audit_report.py              # เทียบ label เดิม/ใหม่, API examples, subgroup checks
│   ├── document.py                  # สร้าง README นี้จาก artifacts และ source ปัจจุบัน
│   └── artifacts/
│       ├── train_test_splits.joblib
│       ├── benchmark_models.joblib
│       ├── model_pipeline.joblib
│       ├── shap_explainer.joblib
│       ├── metadata.json
│       ├── run_manifest.json
│       ├── eda_summary.json
│       ├── test_predictions.csv
│       ├── champion_oof_predictions.csv
│       └── evaluation_curves.json
├── backend/
│   ├── app/                         # FastAPI, Pydantic, inference, scenarios
│   └── requirements.txt
├── frontend/                        # Next.js + React + Tailwind
├── tests/test_pipeline.py           # scientific invariants + integration tests
├── audit/
│   ├── original/                    # source/README/metadata/model/explainer ก่อนแก้
│   ├── original_verification.json
│   ├── label_comparison.json
│   ├── subgroup_metrics.json
│   ├── api_examples.json
│   ├── training.log
│   ├── evaluation.log
│   ├── eda.log
│   ├── tests.log
│   └── frontend-build.log
├── requirements-dev.txt
├── requirements-lock.txt
├── agent-task.md                    # โจทย์/คำสั่งสร้างเดิม ไม่ใช่หลักฐานว่าทำสำเร็จแล้ว
└── README.md
```

directory จริงใช้ `Dataset` ตัว D ใหญ่
ฟังก์ชันค้นหาข้อมูลรองรับทั้ง `Dataset` และ `dataset` เพื่อให้ย้ายเครื่องได้ง่าย
การอ้างอิง path หลักผูกกับตำแหน่งไฟล์ Python ไม่ใช่ current working directory เพียงอย่างเดียว
ข้อมูลดิบไม่ถูก rewrite, drop หรือแทนค่าทับลง CSV
สิ่งที่ clean จะอยู่ใน DataFrame และ artifacts ที่สร้างใหม่

## นิยามโจทย์

หนึ่งแถวหมายถึงผู้เข้าร่วมหนึ่งคน
หนึ่ง sample มี input features 17 ตัว
ผลลัพธ์หลักคือ probability ของ `is_diabetic=1` ภายใต้นิยาม proxy ของโปรเจค
`is_diabetic` เป็น binary target มีค่า 0 หรือ 1
แถวที่หลักฐานไม่พอจะถูกตัดออกจากการสร้าง label

โจทย์นี้เป็น supervised binary classification
ไม่ได้เป็น regression แม้ output จะเป็นทศนิยม 0–1
ไม่ได้เป็น survival analysis เพราะไม่มีเวลาติดตามและเวลาที่เกิดโรค
ไม่ได้จำแนก type 1 กับ type 2 เพราะ label ไม่มีข้อมูลยืนยันชนิดโรคครบถ้วน
ไม่ได้คัดเฉพาะผู้ไม่เคยทราบว่าตนเป็นเบาหวาน
ดังนั้นห้ามตีความผลเป็นโมเดลตรวจ undiagnosed diabetes ที่ผ่านการพิสูจน์แล้ว

คำว่า non-invasive ในระบบหมายถึง **input ตอนกรอกแบบฟอร์มไม่ใช้ผลตรวจเลือด**
การสร้าง target ตอน training ยังต้องอาศัยข้อมูล lab และ diagnosis
การใช้ lab เป็น label ไม่ใช่ leakage โดยตัวมันเอง
จะเป็น leakage หากนำ lab หรือข้อมูลที่นิยาม label ไปเป็นตัวทำนายโดยไม่เหมาะกับ intended use
จึงมีการตรวจไม่ให้ LBXGH, LBXSGL, DIQ010, DIQ050, DIQ070 และยาอยู่ใน FEATURE_COLUMNS

## ข้อมูลดิบและการตรวจแหล่งที่มา

ตรวจพบ SDDSRVYR=8 ใน demographic ทุกแถว
CDC ระบุว่ารหัสนี้เป็น NHANES 2013–2014
จึงระบุรอบสำรวจจากข้อมูลจริง ไม่ใช่เดาจากชื่อแฟ้ม
ดู [CDC DEMO_H](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.htm)

CSV เหล่านี้เป็นตารางที่รวมหลาย NHANES components มาแล้ว
จึงมีหลายร้อยคอลัมน์และบางชื่อมี `.x` / `.y` จากการ merge ในอดีต
การ audit นี้ไม่ได้มีสคริปต์ต้นทางที่แปลง XPT ทั้งหมดเป็น CSV
จึงตรวจความสอดคล้องของตัวแปร/รอบสำรวจและ hash ของไฟล์ที่ให้มาได้
แต่ไม่อ้างว่าพิสูจน์ provenance ทุก byte ย้อนถึงต้นทาง CDC ได้ครบ

| ไฟล์ | แถว | คอลัมน์ | SEQN ไม่ซ้ำ | บทบาท |
| --- | --- | --- | --- | --- |
| demographic.csv | 10175 | 47 | 10175 | ฐานผู้เข้าร่วม อายุ เพศ รอบสำรวจ design variables |
| examination.csv | 9813 | 224 | 9813 | น้ำหนัก ส่วนสูง รอบเอว ความดัน |
| labs.csv | 9813 | 424 | 9813 | HbA1c ใช้สร้าง label; LBXSGL ใช้ audit เท่านั้น |
| diet.csv | 9813 | 168 | 9813 | total nutrients จาก one-day recall และ quality status |
| questionnaire.csv | 10175 | 953 | 10175 | diagnosis, therapy และ lifestyle |
| medications.csv | 20194 | 13 | 10175 | หลายแถวต่อคน ต้อง aggregate ก่อน join |

### ตรวจโครงสร้างก่อนนำข้อมูลเข้าระบบ

`read_table()` ทำงานดังนี้:

1. อ่านเฉพาะคอลัมน์ที่จำเป็นเมื่อกำหนดรายการไว้
2. ใช้ encoding `latin1` เพื่ออ่านชื่อยาโดยไม่ทิ้งตัวอักษรด้วย errors=ignore
3. ตรวจว่าคอลัมน์ที่จำเป็นมีครบ
4. ตรวจว่า SEQN ไม่มี missing
5. สำหรับตารางระดับคน ตรวจว่า SEQN ไม่ซ้ำ
6. สำหรับ medications ยอมให้ซ้ำเพราะหนึ่งคนใช้หลายยาได้

หาก schema หรือ primary key ผิด ระบบ raise error
ไม่ drop_duplicates แบบเงียบ ๆ เพราะอาจซ่อนปัญหา many-to-many join
การบันทึก SHA-256 ทำให้ตรวจได้ว่าข้อมูลที่ใช้รันครั้งถัดไปเป็นไฟล์เดิมหรือไม่
ค่า hash ไม่ใช่การรับรองความถูกต้องของต้นทาง แต่เป็นการระบุชุดข้อมูลที่รัน

| ไฟล์ | จำนวน bytes | SHA-256 |
| --- | --- | --- |
| demographic.csv | 1206856 | 0b261c8cb09b947ed8acfb79b1c10c63bdbf86091c0fd9cb5a930d78e43ad87f |
| diet.csv | 4582708 | 2877e514f96aa731b59404e872ec58aebed7ef894f27f4bf5b6c97e6ac315d12 |
| examination.csv | 4729841 | 8f687a309b8380fad7e5e2eb58550dbc2c7f75864019e717d5839775a6922abd |
| labs.csv | 8322316 | 638d9601cc7041260921338c9ee0cbd40257626392a690019c225f0fc2103930 |
| medications.csv | 1298947 | b5b3542c5f5d14143059bdc56b29fd5111e49b9046c9ca9840e46b5935ff7a12 |
| questionnaire.csv | 12414125 | e175e9361f1049c87fbf15932e55214dbef8a1c8ffb252ebe0ca9d7d5fdcde3f |

## การรวมตารางและคัดเลือกผู้เข้าร่วม

เริ่มจาก demographic เป็นฐาน
เลือก RIDAGEYR >=20 ตาม cohort ที่ต้องการศึกษา
ตัดผู้ที่ RIDEXPRG=1 ซึ่งระบุว่าตั้งครรภ์ออก
ใช้ left join เพื่อไม่ให้คนหายเพียงเพราะไม่มีข้อมูลในบาง component
ทุก join ระดับคนกำหนด `validate='one_to_one'`

ไม่ merge medications แบบตารางดิบเข้า demographic ตรง ๆ
เพราะจะทำให้คนที่ใช้ยาหลายชนิดมีหลายแถวและมีน้ำหนักใน training มากเกินจริง
ใช้ set ของ SEQN ที่พบชื่อยาที่เข้ากฎแทน
จึงรักษาหลักหนึ่งคนต่อหนึ่ง sample

คอลัมน์ RIDEXPRG ไม่ได้มีข้อมูลสำหรับผู้เข้าร่วมทุกคน
รวมถึงกรณีที่ไม่ได้อยู่ในกลุ่มที่ถาม/ตรวจ
จำนวน missing ในตัวแปรนี้จึงไม่ควรตีความว่าเป็นผู้หญิงที่อาจตั้งครรภ์ทั้งหมด
การตัดเฉพาะ known pregnancy ไม่รับประกันว่าคนที่เหลือทุกคนไม่ตั้งครรภ์
ข้อจำกัดนี้ต้องระบุไว้แทนการเดาสถานะ

### จำนวนที่ได้ในแต่ละขั้น

| ขั้นตอน | จำนวนคน |
| --- | --- |
| demographic ทั้งหมด | 10175 |
| อายุ >=20 | 5769 |
| known pregnancy ที่ตัดออก | 65 |
| หลังตัด known pregnancy / หลัง merge | 5704 |
| label ไม่ชัดเจนที่ตัดออก | 442 |
| analytic cohort | 5262 |
| proxy positive | 913 |
| proxy negative | 4349 |

## นิยาม label ที่แก้ไข

### ทำไมไม่ใช้ LBXSGL >=126 ต่อ

LBXSGL เป็น glucose จาก standard biochemistry profile ของ serum
ไม่ได้หมายความว่าทุกค่านี้เป็น fasting plasma glucose ที่ใช้เกณฑ์ 126 ได้
CDC ระบุด้วยว่าผล glucose ใน biochemistry profile ไม่แทน reference-method component
ดู [CDC BIOPRO_H](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BIOPRO_H.htm)

ตรวจ CSV แล้วไม่มีคอลัมน์ `LBXGLU` ซึ่งเป็นตัวแปร fasting plasma glucose ที่คาดหมาย
แม้มี fasting-related fields หลายชื่อ ก็ไม่เพียงพอให้เปลี่ยน LBXSGL เป็น reference fasting assay
มี LBXGLT ของ OGTT แต่ไม่ได้รวมในการนิยาม proxy ฉบับนี้
เราเลือกนิยามที่อธิบายได้ตรงกับหลักฐานที่ใช้ แทนการเพิ่มเกณฑ์ที่ซับซ้อนโดยไม่มี protocol ครบ
ดังนั้น label ใหม่นี้ไม่ใช่ comprehensive clinical diabetes ascertainment

### Positive

กำหนด positive ถ้าเข้าอย่างน้อยหนึ่งข้อ:

- มี HbA1c (`LBXGH`) >=6.5%
- `DIQ010=1`: เคยได้รับการบอกว่ามี diabetes
- `DIQ050=1`: ใช้ insulin อยู่
- `DIQ070=1`: ใช้ยากินลดน้ำตาลอยู่
- พบชื่อยาตาม ingredient matcher และ RXDUSE=1

กฎเป็น OR ไม่ใช่การบวกคะแนน
หนึ่งคนที่เข้า 3 ข้อยังคงเป็น positive หนึ่งคน
positive evidence มีลำดับเหนือ negative evidence
เช่น คนที่ A1c ต่ำเพราะรักษาอยู่แต่มี diagnosis ยังเป็น positive

### Negative

กำหนด negative เมื่อครบทุกข้อ:

- DIQ010=2 คือ explicit no
- มี HbA1c ที่สังเกตจริงและมีค่ามากกว่า 0
- HbA1c <6.5%
- ไม่มี positive evidence ข้างต้น

HbA1c ช่วง 5.7–6.4 ยังเป็น class 0 สำหรับโจทย์ binary diabetes proxy นี้
ไม่ได้แปลว่า healthy หรือไม่มี prediabetes
การแยก prediabetes ต้องสร้าง target และ evaluation ของตนเอง
เกณฑ์ HbA1c และการตรวจยืนยันทางคลินิกอธิบายใน [NIDDK A1C](https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test/)
ผลครั้งเดียวจาก survey ไม่ใช่ clinical confirmation ครบขั้นตอน

### Indeterminate

กรณีที่ไม่มี positive evidence และไม่ผ่าน negative rule จะไม่ถูกฝืนใส่ class 0
ตัวอย่างคือผู้ตอบ no แต่ไม่มี HbA1c
อีกตัวอย่างคือ DIQ010 เป็น refused, don't know หรือ borderline โดยไม่มี positive evidence
ไม่ impute target เพราะจะเป็นการสร้าง ground truth ขึ้นมา

```python
positive = (
    a1c.ge(6.5)
    | df.DIQ010.eq(1)
    | df.DIQ050.eq(1)
    | df.DIQ070.eq(1)
    | df.on_diabetic_med.eq(1)
)
negative = (
    df.DIQ010.eq(2)
    & a1c.notna()
    & a1c.lt(6.5)
    & ~positive
)
# positive -> 1; negative -> 0; ที่เหลือ -> missing label และตัดออก
```

### ยาต้องอ่านจากชื่อสาร ไม่ใช่จากเหตุผลที่ใช้ยา

เดิมมีการค้นคำว่า DIABET ใน RXDRSD1–3
ยาความดันหรือยาอื่นที่ให้กับคนเป็นเบาหวานจึงอาจถูกเรียกว่าเป็นยารักษาเบาหวาน
ฉบับแก้ใช้เฉพาะ RXDDRUG และ ingredient names ที่ระบุไว้ชัดเจน
ใช้ word boundaries เพื่อหลีกเลี่ยงการจับคำบางส่วน
รวม SEQN ที่พบเป็น set ก่อนนำกลับไปสร้างตัวบ่งชี้ระดับคน

รายการชื่อยาในกฎเป็น allowlist ของสารที่เกี่ยวข้องในบริบทข้อมูลรอบนี้
ไม่ได้เป็นฐานข้อมูลยาที่สมบูรณ์สำหรับทุกประเทศหรือทุกปี
metformin อาจใช้ด้วยเหตุผลอื่นได้
ดังนั้น medication-only positive ยังคงเป็น proxy ไม่ใช่การยืนยันชนิดโรค
DIQ050/DIQ070 เป็นหลักฐานเสริมจากคำถามเฉพาะการรักษา diabetes
ดู [CDC DIQ_H](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DIQ_H.htm)

### ผลเปรียบเทียบ label เดิมกับใหม่

| รายการ | จำนวน |
| --- | --- |
| original_labeled | 5654 |
| original_positive | 1095 |
| corrected_labeled | 5262 |
| corrected_positive | 913 |
| common_subjects | 5261 |
| common_old_positive_new_negative | 170 |
| common_old_negative_new_positive | 0 |
| old_labeled_now_excluded | 393 |
| newly_included | 1 |
| medication_parser_old_subjects | 685 |
| medication_parser_new_subjects | 678 |
| old_only_medication_subjects | 7 |

ผู้เข้าร่วมร่วมกันที่เปลี่ยนจาก positive เดิมเป็น negative ใหม่มี 170 คน
ไม่ควรเรียกคนเหล่านี้ว่า “ยืนยันว่าไม่เป็นเบาหวาน”
เพียงแต่ label เดิมกับ proxy ใหม่ใช้หลักฐานต่างกัน
มี 176 คนใน cohort หลังตัด known pregnancy ที่ LBXSGL>=126 แต่ไม่เข้า positive rule ใหม่
จำนวนนี้ไม่เท่ากับจำนวนคนที่เปลี่ยน label เสมอ เพราะบางคนถูกตัดเป็น indeterminate
หลักฐาน positive แต่ละชนิดด้านล่างนับทับซ้อนกันได้

| positive evidence | จำนวนผู้เข้าเงื่อนไขก่อนตัด label ไม่ชัด |
| --- | --- |
| a1c_positive | 599 |
| diagnosis_positive | 720 |
| insulin_positive | 209 |
| pills_positive | 544 |
| medication_positive | 661 |

## Cleaning และ feature engineering

### สิ่งที่เป็น deterministic กับสิ่งที่ต้อง fit

การแปลงหน่วย การ map questionnaire code และการหารน้ำหนักด้วยส่วนสูงกำลังสอง
เป็นกฎที่กำหนดล่วงหน้า ไม่ต้องเรียนจากข้อมูล
ทำก่อน split ได้ตราบใดที่ไม่ใช้ distribution หรือ target ของ test มาตัดสินกฎ
ส่วน median, mode, center และ scale ต้องเรียนจาก training เท่านั้น
จึงวางอยู่ใน sklearn Pipeline ภายใน CV

### Missing ไม่ใช่ศูนย์

น้ำหนัก ส่วนสูง รอบเอว และ HbA1c ที่ไม่เป็นบวกไม่ถูกนำไปคำนวณตรง ๆ
questionnaire binary ยอมรับเฉพาะ code 1/2
refused/don't know กลายเป็น NaN
diet ที่ไม่ผ่าน quality status กลายเป็น missing
ไฟเบอร์ 0 เป็นค่าปริมาณที่อาจเกิดขึ้นจริง แต่ ratio sugar/fiber ณ จุดนั้นไม่มีนิยาม
จึงทำให้เฉพาะ ratio เป็น missing โดยไม่เปลี่ยน fiber=0 เป็นค่าบวกปลอม

### ไม่มีการลบ outlier ด้วย target หรือ test distribution

โปรเจคนี้ไม่ได้ลบค่าทุกตัวที่อยู่นอก 1.5 IQR แบบอัตโนมัติ
ไม่ได้ winsorize จากทั้ง cohort ก่อน CV
ค่าที่สุดโต่งแต่เป็นบวกและอยู่ในข้อมูลยังอยู่ใน training
RobustScaler ช่วยเรื่องการแปลงสเกล แต่ไม่ได้ทำให้ค่าที่ผิดหรือ extreme หายไป
การพิสูจน์ว่า outlier เป็น measurement error ต้องใช้ข้อมูลต้นทางและ protocol เพิ่ม

### BMI และอัตราส่วนรอบเอว

```python
df['bmi'] = df.weight_kg / (df.height_cm / 100).pow(2)
df['waist_to_height_ratio'] = df.waist_cm / df.height_cm
# ตัวอย่าง 90 kg / 1.72^2 = 30.4219... kg/m²
# train และ serving ใช้สูตรเดียวกัน; ปัดเศษเฉพาะตอนแสดงผล
```

ใช้ BMI ที่คำนวณจากน้ำหนักและส่วนสูงโดยตรงทั้ง train และ serving
จึงไม่ใช้ BMI จาก CSV ที่ปัดเศษแล้วเฉพาะ training
ไม่ใช้สูตรเดารอบเอวจากส่วนสูงเมื่อผู้ใช้ไม่กรอก
หากไม่มีรอบเอว ทั้ง waist_cm และ waist_to_height_ratio จะ missing
model pipeline เติมด้วย median ของ features เหล่านั้นตามที่เรียนจาก fit
UI แสดงว่าไม่มีค่าที่วัดจริง แทนการโชว์ค่าเดาเสมือนวัดแล้ว

### ความดันโลหิต

ใช้ค่าเฉลี่ยของ BPXSY1–3 ที่มีค่า >0 สำหรับ systolic
ใช้ค่าเฉลี่ยของ BPXDI1–3 ที่มีค่า >0 สำหรับ diastolic
ถ้าไม่มี reading ใช้งานได้จะเป็น missing
การใช้ค่าเฉลี่ยช่วยลดความผันผวนจากการหยิบ reading แรกเพียงค่าเดียว
นี่เป็น operational definition ของโปรเจค ไม่ใช่การทำซ้ำทุกกรณีพิเศษใน clinical BP protocol
การถือ zero diastolic ว่าใช้ไม่ได้เป็นตัวเลือก conservative ของงานนี้และควรตรวจละเอียดหากใช้จริง
ค่าจากแบบฟอร์มใช้เป็นหนึ่ง summary reading ที่ผู้ใช้ให้ ไม่ได้จำลองว่ามีการวัดสามครั้ง

hypertension_flag=1 หากมีประวัติ BPQ020=1 หรือ systolic>=140 หรือ diastolic>=90
เกณฑ์นี้เป็น feature engineering ที่คงไว้ตามโจทย์เดิม ไม่ใช่การประกาศเกณฑ์วินิจฉัยความดันปัจจุบัน
ถ้า history unknown และไม่มี BP ที่เข้า positive rule จะเก็บ unknown
การวัดความดันปกติครั้งเดียวไม่ได้พิสูจน์ว่าไม่เคยมีประวัติความดันสูง

### อาหาร

ใช้ total nutrients จาก recall วันแรก
DR1DRSTZ=1 หมายถึง reliable and complete ตาม codebook
ดู [CDC DR1TOT_H](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DR1TOT_H.htm)
DR1TSUGR คือ total sugar ไม่ใช่ added sugar
จึงไม่เอาค่านี้ไปเทียบกับเพดาน added sugar แล้วสรุปว่าผู้ใช้กินน้ำตาลเติมเกินเกณฑ์
หนึ่งวันไม่เท่ากับ usual intake และมี recall/reporting error

### กิจกรรม สูบบุหรี่ และการนอน

PAQ605 เดิมเป็น vigorous work activity
การกดว่าออกกำลังกาย 30 นาทีจึงไม่ได้สอดคล้องกับตัวแปร training เดิม
ฉบับนี้เปลี่ยนเป็น PAQ650 ซึ่งถาม vigorous recreational activity
UI ถามกิจกรรมหนักต่อเนื่องอย่างน้อย 10 นาทีใน typical week ตามบริบทคำถาม
ดู [CDC PAQ_H](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAQ_H.htm)
ไม่มีกิจกรรมหนักเพื่อสันทนาการไม่ได้แปลว่าไม่ได้เดิน ไม่ทำงานหนัก หรือ sedentary ทั้งวัน
SMQ020 คือเคยสูบบุหรี่รวมอย่างน้อย 100 มวน ไม่ใช่ current smoker
SLQ050 คือเคยบอกแพทย์เรื่อง trouble sleeping ไม่ใช่การวินิจฉัย insomnia ทุกกรณี

## พจนานุกรม features ทั้งหมด

### `age`

- ตัวแปรต้นทาง: `RIDAGEYR`
- หน่วย/ชนิด: ปี; อายุ 20+ และ 80 เป็น top-code
- กฎคำนวณ: `clip(upper=80)`
- ข้อควรเข้าใจ: อายุจริงตั้งแต่ 80 ขึ้นไปไม่สามารถแยกได้จากข้อมูลนี้
- Median ของ fit partition: 48
- Mean ของ fit partition: 48.9651
- ช่วงที่สังเกตใน fit: 20 ถึง 80
- จำนวน missing ใน fit: 0 แถว (0.00%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `bmi`

- ตัวแปรต้นทาง: `BMXWT, BMXHT`
- หน่วย/ชนิด: kg/m²
- กฎคำนวณ: `weight_kg / (height_cm/100)^2`
- ข้อควรเข้าใจ: มีความสัมพันธ์ซ้ำกับน้ำหนักและส่วนสูง
- Median ของ fit partition: 27.706
- Mean ของ fit partition: 28.9189
- ช่วงที่สังเกตใน fit: 14.1408 ถึง 65.4999
- จำนวน missing ใน fit: 46 แถว (1.46%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `waist_cm`

- ตัวแปรต้นทาง: `BMXWAIST`
- หน่วย/ชนิด: cm
- กฎคำนวณ: `ใช้ค่าบวกที่สังเกตจริง`
- ข้อควรเข้าใจ: หายแล้ว impute; ไม่มีสูตรเดารอบเอว
- Median ของ fit partition: 97.2
- Mean ของ fit partition: 98.7898
- ช่วงที่สังเกตใน fit: 55.5 ถึง 165.5
- จำนวน missing ใน fit: 170 แถว (5.39%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `height_cm`

- ตัวแปรต้นทาง: `BMXHT`
- หน่วย/ชนิด: cm
- กฎคำนวณ: `ใช้ค่าบวกที่สังเกตจริง`
- ข้อควรเข้าใจ: เป็นตัวหารของ BMI และ waist/height
- Median ของ fit partition: 166.9
- Mean ของ fit partition: 167.253
- ช่วงที่สังเกตใน fit: 136.3 ถึง 202.6
- จำนวน missing ใน fit: 43 แถว (1.36%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `weight_kg`

- ตัวแปรต้นทาง: `BMXWT`
- หน่วย/ชนิด: kg
- กฎคำนวณ: `ใช้ค่าบวกที่สังเกตจริง`
- ข้อควรเข้าใจ: what-if คำนวณ BMI ใหม่ด้วยน้ำหนักนี้
- Median ของ fit partition: 78
- Mean ของ fit partition: 81.1348
- ช่วงที่สังเกตใน fit: 32.8 ถึง 193.6
- จำนวน missing ใน fit: 40 แถว (1.27%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `waist_to_height_ratio`

- ตัวแปรต้นทาง: `BMXWAIST, BMXHT`
- หน่วย/ชนิด: ไม่มีหน่วย
- กฎคำนวณ: `waist_cm / height_cm`
- ข้อควรเข้าใจ: ไม่ใช้เป็นเหตุผลว่าลดเอวแล้วโรคลดลงเท่า SHAP
- Median ของ fit partition: 0.582625
- Mean ของ fit partition: 0.59139
- ช่วงที่สังเกตใน fit: 0.364412 ถึง 1.02731
- จำนวน missing ใน fit: 175 แถว (5.54%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `systolic_bp`

- ตัวแปรต้นทาง: `BPXSY1–3`
- หน่วย/ชนิด: mmHg
- กฎคำนวณ: `mean ของ readings >0`
- ข้อควรเข้าใจ: missing เมื่อไม่มี reading ใช้ได้
- Median ของ fit partition: 120
- Mean ของ fit partition: 123.284
- ช่วงที่สังเกตใน fit: 73.3333 ถึง 228
- จำนวน missing ใน fit: 115 แถว (3.64%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `diastolic_bp`

- ตัวแปรต้นทาง: `BPXDI1–3`
- หน่วย/ชนิด: mmHg
- กฎคำนวณ: `mean ของ readings >0`
- ข้อควรเข้าใจ: zero readings ถูกเว้นออกตามนิยามโปรเจค
- Median ของ fit partition: 70.3333
- Mean ของ fit partition: 70.1485
- ช่วงที่สังเกตใน fit: 4 ถึง 116.667
- จำนวน missing ใน fit: 126 แถว (3.99%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `daily_sugar_g`

- ตัวแปรต้นทาง: `DR1TSUGR`
- หน่วย/ชนิด: g จาก recall วันแรก
- กฎคำนวณ: `ใช้เมื่อ DR1DRSTZ=1 และค่า>=0`
- ข้อควรเข้าใจ: เป็น total sugar ไม่ใช่ added sugar
- Median ของ fit partition: 90.85
- Mean ของ fit partition: 108.846
- ช่วงที่สังเกตใน fit: 0.13 ถึง 979.39
- จำนวน missing ใน fit: 304 แถว (9.63%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `daily_fiber_g`

- ตัวแปรต้นทาง: `DR1TFIBE`
- หน่วย/ชนิด: g จาก recall วันแรก
- กฎคำนวณ: `ใช้เมื่อ DR1DRSTZ=1 และค่า>=0`
- ข้อควรเข้าใจ: fiber=0 ยังเก็บ 0; ratio จะ missing
- Median ของ fit partition: 14.7
- Mean ของ fit partition: 16.9296
- ช่วงที่สังเกตใน fit: 0 ถึง 107
- จำนวน missing ใน fit: 304 แถว (9.63%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `daily_calories_kcal`

- ตัวแปรต้นทาง: `DR1TKCAL`
- หน่วย/ชนิด: kcal จาก recall วันแรก
- กฎคำนวณ: `ใช้เมื่อ DR1DRSTZ=1 และค่า>=0`
- ข้อควรเข้าใจ: ถ้า UI ไม่ส่งค่า model จะ impute อย่างเปิดเผย
- Median ของ fit partition: 1914.5
- Mean ของ fit partition: 2114.54
- ช่วงที่สังเกตใน fit: 193 ถึง 10293
- จำนวน missing ใน fit: 304 แถว (9.63%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `sugar_to_fiber_ratio`

- ตัวแปรต้นทาง: `sugar/fiber`
- หน่วย/ชนิด: อัตราส่วน
- กฎคำนวณ: `sugar / fiber เฉพาะ fiber>0`
- ข้อควรเข้าใจ: ไม่มีการบวก epsilon ที่ทำให้เกิดค่าหลายล้าน
- Median ของ fit partition: 5.88033
- Mean ของ fit partition: 9.14406
- ช่วงที่สังเกตใน fit: 0.05375 ถึง 661.2
- จำนวน missing ใน fit: 307 แถว (9.73%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `gender_is_female`

- ตัวแปรต้นทาง: `RIAGENDR`
- หน่วย/ชนิด: binary
- กฎคำนวณ: `1→0, 2→1`
- ข้อควรเข้าใจ: สะท้อน coding ของ survey; API ต้องระบุ male/female
- Median ของ fit partition: 1
- Mean ของ fit partition: 0.516477
- ช่วงที่สังเกตใน fit: 0 ถึง 1
- จำนวน missing ใน fit: 0 แถว (0.00%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `hypertension_flag`

- ตัวแปรต้นทาง: `BPQ020 และ BP summary`
- หน่วย/ชนิด: binary หรือ missing
- กฎคำนวณ: `history yes OR systolic>=140 OR diastolic>=90`
- ข้อควรเข้าใจ: คง history เมื่อจำลองว่าความดันปัจจุบันลดลง
- Median ของ fit partition: 0
- Mean ของ fit partition: 0.420285
- ช่วงที่สังเกตใน fit: 0 ถึง 1
- จำนวน missing ใน fit: 1 แถว (0.03%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `no_vigorous_recreation_flag`

- ตัวแปรต้นทาง: `PAQ650`
- หน่วย/ชนิด: binary หรือ missing
- กฎคำนวณ: `yes→0, no→1`
- ข้อควรเข้าใจ: ห้ามเรียกเป็น sedentary flag
- Median ของ fit partition: 1
- Mean ของ fit partition: 0.783904
- ช่วงที่สังเกตใน fit: 0 ถึง 1
- จำนวน missing ใน fit: 0 แถว (0.00%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `smoking_history_flag`

- ตัวแปรต้นทาง: `SMQ020`
- หน่วย/ชนิด: binary หรือ missing
- กฎคำนวณ: `yes→1, no→0`
- ข้อควรเข้าใจ: ประวัติ 100 มวน ไม่ใช่สถานะสูบปัจจุบัน
- Median ของ fit partition: 0
- Mean ของ fit partition: 0.437401
- ช่วงที่สังเกตใน fit: 0 ถึง 1
- จำนวน missing ใน fit: 1 แถว (0.03%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

### `sleep_trouble_flag`

- ตัวแปรต้นทาง: `SLQ050`
- หน่วย/ชนิด: binary หรือ missing
- กฎคำนวณ: `yes→1, no→0`
- ข้อควรเข้าใจ: ประวัติเคยบอกแพทย์เรื่องการนอน
- Median ของ fit partition: 0
- Mean ของ fit partition: 0.262357
- ช่วงที่สังเกตใน fit: 0 ถึง 1
- จำนวน missing ใน fit: 0 แถว (0.00%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง

## การแบ่งข้อมูลและป้องกัน leakage

### บทบาทของแต่ละ partition

แบ่ง test=20% ออกก่อนด้วย stratification ตาม target
จาก development=80% แบ่ง calibration=25% ของ development
จึงได้สัดส่วนรวมโดยประมาณ fit 60% / calibration 20% / test 20%
ทั้งสองครั้งใช้ random_state=42
จำนวนจริงอาจไม่ตรงเปอร์เซ็นต์พอดีเพราะต้องเป็นจำนวนเต็ม
ใช้ SEQN เป็น index และตรวจ set ของแต่ละ partition ว่าไม่ทับกัน

| partition | n | positive | negative | positive fraction |
| --- | --- | --- | --- | --- |
| fit | 3156 | 547 | 2609 | 17.3321% |
| calibration | 1053 | 183 | 870 | 17.3789% |
| test | 1053 | 183 | 870 | 17.3789% |

```python
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
)
```

fit ใช้สร้างโมเดล ทำ CV เลือก champion และสร้าง OOF responses
calibration ใช้เรียน sigmoid map หลัง base model ถูก freeze แล้ว
ไม่มีการใช้ test เลือก hyperparameters, champion หรือ screening threshold
test ใช้รายงาน final performance ของ pipeline ที่กำหนดไว้
โมเดลที่ export จึงยังมี base estimator ซึ่ง fit บน 60% ไม่ใช่ fit ใหม่บนทั้ง cohort
การ refit หลังเห็น test จะสร้างโมเดลคนละตัวและต้องประเมินใหม่

### ภายในแต่ละ CV fold

แบ่ง fit partition เป็น 5 folds แบบ stratified และ shuffle=True
ในแต่ละรอบ ใช้ 4 folds เป็น fold-training และอีก 1 fold เป็น validation
สร้าง estimator และ preprocessor ใหม่ด้วย clone
fit median/mode/scaler เฉพาะ fold-training
transform validation ด้วยค่าที่เรียนมาแล้ว
fit classifier เฉพาะ fold-training
ทำนาย validation แล้วบันทึก prediction ไว้ตำแหน่งเดิม
ท้ายสุดทุกคนใน fit ได้หนึ่ง out-of-fold prediction

ไม่มีการ fit preprocessor บน X_fit ทั้งชุดก่อนแบ่ง folds
นี่คือสิ่งที่แก้จากโค้ดเดิมโดยตรง

```python
model = Pipeline([
    ('preprocessor', build_preprocessor()),
    ('classifier', clone(classifier)),
])
model.fit(X_fit.iloc[train_idx], y_fit.iloc[train_idx])
p = model.predict_proba(X_fit.iloc[val_idx])[:, 1]
```

### Imputer และ scaler ทำอะไร

numeric features ใช้ SimpleImputer(strategy='median')
จากนั้น RobustScaler ลบ center และหารด้วยสเกลที่อาศัย interquartile range
binary features ใช้ SimpleImputer(strategy='most_frequent') และไม่ scale
keep_empty_features=True รักษาจำนวนคอลัมน์หากบาง fold ไม่มีค่าที่สังเกตได้เลย
ColumnTransformer บังคับลำดับคอลัมน์ชัดเจน

ต้นไม้โดยมากไม่จำเป็นต้อง scale
การใช้ preprocessor เดียวกันใน benchmark เป็นทางเลือกเพื่อความสม่ำเสมอและรองรับ Logistic Regression
ไม่ได้อ้างว่า RobustScaler จะทำให้ Random Forest แม่นขึ้นโดยอัตโนมัติ

### Leakage ที่ป้องกันแล้วและข้อจำกัดที่เหลือ

ไม่ใช้ SEQN เป็น predictor
ไม่ใช้ target-defining labs/diagnosis/medication เป็น predictor
ไม่เรียน imputation/scaling จาก validation/test
ไม่ใช้ test median เป็น fallback ของ API
ไม่ใช้ test scores ตั้ง percentile reference
ไม่ใช้ test labels เลือก champion หรือ threshold

อย่างไรก็ตาม NHANES เป็น clustered survey
random split รายคนอาจให้ผู้เข้าร่วมจาก survey clusters เดียวกันกระจายอยู่หลาย partition
SDMVPSU เป็น masked PSU code ที่ต้องตีความร่วมกับ strata ไม่ใช่ household ID
ฉบับนี้ไม่ได้สร้าง independent-geography หรือ household-grouped validation
จึงจำกัดข้ออ้างไว้ที่ internal unweighted participant-level benchmark
อีกทั้งผล test ของรุ่นเดิมเคยถูกเปิดดูแล้ว จึงไม่ใช่ pristine prospective holdout ของโครงการวิจัยใหม่
ควรมี external cohort หรือ temporal validation ก่อนอ้าง generalization จริง

## EDA บนชุดฝึก

EDA ฉบับใหม่อ่าน `train_test_splits.joblib` แล้วใช้เฉพาะ X_train/y_train ซึ่งคือ fit partition
เพื่อหลีกเลี่ยงการเลือก features จากความสัมพันธ์กับ test target
ไฟล์ `eda_summary.json` ระบุ scope ชัดเจน
EDA ที่รายงานนี้เป็น descriptive ไม่ใช่ผลสำรวจประชากรสหรัฐฯ แบบถ่วงน้ำหนัก

| feature | missing % ใน fit | Pearson correlation กับ proxy ใน fit |
| --- | --- | --- |
| age | 0.00 | +0.3078 |
| bmi | 1.46 | +0.2183 |
| waist_cm | 5.39 | +0.2916 |
| height_cm | 1.36 | -0.0102 |
| weight_kg | 1.27 | +0.1942 |
| waist_to_height_ratio | 5.54 | +0.2929 |
| systolic_bp | 3.64 | +0.2055 |
| diastolic_bp | 3.99 | -0.0167 |
| daily_sugar_g | 9.63 | -0.1110 |
| daily_fiber_g | 9.63 | +0.0016 |
| daily_calories_kcal | 9.63 | -0.0992 |
| sugar_to_fiber_ratio | 9.73 | -0.0321 |
| gender_is_female | 0.00 | -0.0444 |
| hypertension_flag | 0.03 | +0.2725 |
| no_vigorous_recreation_flag | 0.00 | +0.1367 |
| smoking_history_flag | 0.03 | +0.0519 |
| sleep_trouble_flag | 0.00 | +0.1170 |

### ความหมายของ correlation

Pearson correlation แสดงความสัมพันธ์เชิงเส้นกับ binary target ตามข้อมูลที่มี
ไม่ได้พิสูจน์สาเหตุ
ไม่ได้เป็น p-value และไม่ได้แก้ confounding
การมี correlation ใกล้ศูนย์ไม่ได้พิสูจน์ว่า feature ไม่มีประโยชน์ใน nonlinear model
การมีหลาย features ที่สัมพันธ์กันทำให้ importance กระจายและตีความแยกยาก
BMI, weight, height, waist และ waist/height มีความสัมพันธ์ทางสูตรโดยตรง

ถ้า sugar หรือ calories มีความสัมพันธ์เชิงลบกับ diabetes proxy
ห้ามสรุปว่ากินน้ำตาลเพิ่มช่วยป้องกัน diabetes
ผู้ที่ได้รับการวินิจฉัยแล้วอาจเปลี่ยนพฤติกรรมการกิน
นี่คือ reverse causation ที่เกิดได้ใน cross-sectional data
การอธิบาย SHAP เดิมว่า sugar สูงทำให้ score สูงเสมอจึงไม่สอดคล้องกับโมเดลทุกกรณี

## โมเดลและ cross-validation

ใช้ candidate set คงที่ 6 รายการรวม baseline
ไม่มี GridSearchCV, RandomizedSearchCV หรือการค้น hyperparameter อื่นในฉบับนี้
ค่าพารามิเตอร์ระบุไว้ก่อนประเมิน และ export ได้จากโค้ด train.py
จึงเรียกขั้นตอนนี้ว่า fixed-configuration model benchmarking
ไม่เขียนว่า “ทำ hyperparameter tuning แล้ว” เพราะไม่มีหลักฐานว่าเกิดขึ้น
LightGBM/XGBoost ไม่อยู่ใน benchmark ที่แก้ไข แม้บาง package จะติดตั้งอยู่ใน environment เดิม
การเพิ่มโมเดลตาม import สำเร็จแบบเงียบ ๆ ทำให้ชุดทดลองเปลี่ยนข้ามเครื่องได้ จึงเอาพฤติกรรมนั้นออก

### ตาราง CV — base model ก่อน calibration

ค่า ROC-AUC/AP ในตารางนี้มาจาก validation folds ของ fit partition เท่านั้น
แสดง mean ± sample standard deviation (ddof=1) ของ 5 folds
recall และ F1 ใน CV ใช้ threshold=0.5 ของ base model จึงไม่ใช่ operating threshold ที่เลือกภายหลัง
ไม่ได้เป็น nested-CV estimate ของกระบวนการเลือกโมเดลทั้งหมด
คะแนน CV ของผู้ชนะอาจมี selection optimism; final test และ external validation มีบทบาทต่างหาก

| โมเดล | CV ROC-AUC mean ± SD | CV AP mean ± SD | CV recall@0.5 | CV F1@0.5 |
| --- | --- | --- | --- | --- |
| DummyPrior | 0.5000 ± 0.0000 | 0.1733 ± 0.0008 | 0.0000 | 0.0000 |
| LogisticRegression | 0.8013 ± 0.0178 | 0.4471 ± 0.0397 | 0.7568 | 0.4788 |
| RandomForest | 0.8072 ± 0.0192 | 0.4610 ± 0.0252 | 0.7077 | 0.4763 |
| ExtraTrees | 0.8033 ± 0.0171 | 0.4603 ± 0.0254 | 0.7294 | 0.4720 |
| GradientBoosting | 0.8037 ± 0.0148 | 0.4555 ± 0.0219 | 0.2120 | 0.2988 |
| HistGradientBoosting | 0.7977 ± 0.0152 | 0.4491 ± 0.0248 | 0.6438 | 0.4671 |

เลือก **RandomForest** เพราะมี mean CV ROC-AUC สูงสุดในกลุ่มโมเดลที่ใช้ features
DummyPrior เป็น baseline จึงไม่ถูกเลือกเป็น champion
การตัดสินใช้ค่าทศนิยมเต็มจาก artifact ไม่ใช่ตัวเลขที่ปัดเศษในตาราง
ส่วนต่าง CV ที่เล็กไม่เพียงพอจะยืนยันว่าชนะเชิงสถิติ
ไม่มีการอ้างว่า Random Forest ดีกว่าทุกโมเดลในทุก population

## Calibration และ threshold

### ทำไมต้อง calibrate

ROC-AUC วัด ranking ไม่ได้ยืนยันว่าคะแนน 0.4 เท่ากับโอกาส 40% ที่เชื่อถือได้
class_weight='balanced' เปลี่ยน objective และอาจทำให้ raw probability ไม่ตรง observed frequency
ข้อมูลชุดนี้มี class imbalance จึงต้องตรวจ calibration แยกจาก discrimination

ใช้ `CalibratedClassifierCV(FrozenEstimator(raw), method='sigmoid', ensemble=False)`
raw model ถูก fit บน fit partition ไปแล้ว
FrozenEstimator ป้องกันการ refit base model บน calibration partition
sigmoid map เรียนจาก X_cal/y_cal โดยใช้ outcome ของ calibration เท่านั้น
scikit-learn ใช้ decision_function หากโมเดลมี เช่น Logistic Regression
ถ้าไม่มีจึงใช้ predict_proba ของ positive class
ดู [CalibratedClassifierCV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)

```python
calibrated = CalibratedClassifierCV(
    FrozenEstimator(raw), method='sigmoid', ensemble=False
)
calibrated.fit(X_cal, y_cal)
p_test = calibrated.predict_proba(X_test)[:, 1]
```

การมี calibration layer ไม่ได้เป็นหลักประกันว่าโมเดล calibrated สมบูรณ์
ต้องดู Brier/log loss และ reliability curve บนข้อมูลที่ไม่ใช้เรียน calibration
ห้ามใช้ accuracy แทนการตรวจ calibration
การประเมินนี้เก็บทั้ง raw probability และ calibrated probability เพื่อตรวจความแตกต่าง

### กฎเลือก screening threshold

ตั้งเป้าหมายเชิงการศึกษาให้ sensitivity บน fit OOF อย่างน้อย 0.80
นี่เป็น operating preference ของโปรเจค ไม่ใช่เกณฑ์ทางคลินิกที่ผ่านการรับรอง
OOF responses มาจาก models ที่ไม่เคย train บน sample นั้น
แปลง response เหล่านี้ผ่าน sigmoid map ที่เรียนจาก calibration partition แยกกัน
จากนั้นดู ROC curve ของ fit OOF mapped scores
เลือก threshold ที่ specificity สูงที่สุดภายใต้ sensitivity>=0.80
หากเท่ากันเลือก threshold สูงกว่า

ไม่เลือก threshold จาก ROC curve ของ test
ไม่ calibrate ด้วย test
และไม่เปลี่ยน threshold เพื่อให้ subgroup test ดูดีขึ้น
ค่า sensitivity ของ OOF ณ threshold ที่เลือกยังเป็น development estimate
เพราะ threshold ถูก optimize จาก OOF labels ชุดนั้น
และ OOF base models ฝึกบนข้อมูลน้อยกว่า final base model
ผล final test จึงอาจไม่ตรงกับ development sensitivity เป๊ะ

```python
fpr, tpr, thresholds = roc_curve(y_fit, mapped_oof, drop_intermediate=False)
valid = np.flatnonzero((tpr >= 0.80) & np.isfinite(thresholds))
i = min(valid, key=lambda k: (fpr[k], -thresholds[k]))
threshold = float(thresholds[i])
y_pred = (p_test >= threshold).astype(int)
```

threshold ของ champion = **0.123924534734** บนสเกล 0–1
หรือประมาณ **12.392%** บนหน้าจอ
ไม่ใช่ 50% และไม่ใช่ค่า 20% ที่ใช้เป็นขอบสีของหน้าปัด

สีแบ่งเป็น lower (<20%), intermediate (20–50%), higher (>50%)
ช่วงสีเหล่านี้เป็น display bands ที่ยังไม่ผ่าน clinical validation
ผล `screening_positive` ใช้ threshold ที่เลือกจริง แยกออกจาก `risk_tier`
คนที่มี score 15% จึงอาจอยู่ lower color band แต่เกิน screening threshold ได้
UI แสดง threshold แยกชัดเจนเพื่อไม่ให้ใช้สีแทนการตัดสิน

## ผลทดสอบของทุกโมเดล

### นิยาม metrics ก่อนอ่านตัวเลข

TP: proxy positive และ model positive
FN: proxy positive แต่ model negative
TN: proxy negative และ model negative
FP: proxy negative แต่ model positive

- Sensitivity/Recall = TP / (TP+FN)
- Specificity = TN / (TN+FP)
- Precision/PPV = TP / (TP+FP)
- F1 = 2 × Precision × Recall / (Precision+Recall)
- Accuracy = (TP+TN) / N
- ROC-AUC = ความสามารถจัดอันดับ positive สูงกว่า negative โดยพิจารณาทุก threshold
- Average Precision = สรุป precision-recall แบบ AP ของ sklearn
- Brier = mean((probability - observed_label)^2), ต่ำดีกว่า
- Log loss = ค่า negative log-likelihood เฉลี่ย, ต่ำดีกว่า

คำว่า PR-AUC ใน key `test_pr_auc` เป็น legacy name ที่คงไว้ให้ frontend อ่านได้
ค่าจริงคำนวณด้วย `average_precision_score`
ไม่ได้ใช้ trapezoidal integration ของ precision-recall curve
จึงเรียก AP ในตารางและข้อความเพื่อไม่สับสน
ดู [sklearn average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

### ตาราง test ของ calibrated models ณ threshold ที่เลือกจาก development

ทุกโมเดลใช้ test subjects ชุดเดียวกัน
ค่า threshold ต่างกันเพราะเลือกจาก OOF ของแต่ละโมเดล
ตัวเลขในตารางนี้ไม่ได้ใช้เลือก champion

| โมเดล | threshold | Test AUC | Test AP | Sensitivity | Specificity | Precision | F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DummyPrior | 0.173797 | 0.5000 | 0.1738 | 1.0000 | 0.0000 | 0.1738 | 0.2961 | 0.1738 |
| LogisticRegression | 0.138589 | 0.8086 | 0.4006 | 0.8415 | 0.6437 | 0.3319 | 0.4760 | 0.6781 |
| RandomForest | 0.123925 | 0.8161 | 0.4484 | 0.8579 | 0.6011 | 0.3115 | 0.4571 | 0.6458 |
| ExtraTrees | 0.133950 | 0.8030 | 0.4177 | 0.8087 | 0.6356 | 0.3183 | 0.4568 | 0.6657 |
| GradientBoosting | 0.142102 | 0.8184 | 0.4541 | 0.8634 | 0.6253 | 0.3264 | 0.4738 | 0.6667 |
| HistGradientBoosting | 0.147892 | 0.8056 | 0.4336 | 0.8306 | 0.6184 | 0.3140 | 0.4558 | 0.6553 |

### ตาราง test ที่ threshold=0.5 เพื่อดูผลของ operating point

โมเดลอาจมี AUC ใกล้เคียงกันแต่ recall ต่างกันมากเมื่อใช้ threshold เดียวกัน
การเลือก threshold เป็นส่วนสำคัญของ screening workflow
การเพิ่ม recall มักแลกกับจำนวน FP ที่สูงขึ้น
ไม่มีการบังคับให้ตัวเลข test recall ต้องเกิน 80% ทุกโมเดลหรือทุกกลุ่ม

| โมเดล | Recall@0.5 | Specificity@0.5 | Precision@0.5 | F1@0.5 | Accuracy@0.5 |
| --- | --- | --- | --- | --- | --- |
| DummyPrior | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8262 |
| LogisticRegression | 0.1311 | 0.9759 | 0.5333 | 0.2105 | 0.8291 |
| RandomForest | 0.1639 | 0.9747 | 0.5769 | 0.2553 | 0.8338 |
| ExtraTrees | 0.1093 | 0.9874 | 0.6452 | 0.1869 | 0.8348 |
| GradientBoosting | 0.1421 | 0.9770 | 0.5652 | 0.2271 | 0.8319 |
| HistGradientBoosting | 0.0820 | 0.9851 | 0.5357 | 0.1422 | 0.8281 |

### Calibration บน final test: raw เทียบกับ calibrated

Brier และ log loss คำนวณจาก probabilities ไม่ขึ้นกับ classification threshold
การเปลี่ยน threshold ไม่สามารถทำให้ Brier ดีขึ้นได้
ผล calibration ไม่จำเป็นต้องทำให้ ROC-AUC ดีขึ้น เพราะ sigmoid ที่ monotonic มักรักษา ranking

| โมเดล | Raw Brier | Calibrated Brier | Raw log loss | Calibrated log loss |
| --- | --- | --- | --- | --- |
| DummyPrior | 0.14359 | 0.14359 | 0.46184 | 0.46184 |
| LogisticRegression | 0.19139 | 0.12100 | 0.56285 | 0.37367 |
| RandomForest | 0.16803 | 0.11622 | 0.49436 | 0.36541 |
| ExtraTrees | 0.18299 | 0.12021 | 0.53617 | 0.37355 |
| GradientBoosting | 0.11594 | 0.11601 | 0.36380 | 0.36554 |
| HistGradientBoosting | 0.16174 | 0.11854 | 0.47171 | 0.37101 |

### วิเคราะห์ `DummyPrior` รายตัว

กลไก: คืนค่า class prior โดยไม่ใช้ features

พารามิเตอร์หลัก: เป็น baseline ว่าใช้ ML แล้วดีกว่าการทาย prevalence คงที่หรือไม่

ข้อควรเข้าใจ: AUC=0.5 และ AP ใกล้ positive fraction; threshold ที่บังคับ recall>=80% ทำให้ทาย positive ทั้งหมดได้

ผล CV ก่อน calibration:

- ROC-AUC mean = 0.500000
- ROC-AUC SD = 0.000000
- Average Precision mean = 0.173320
- Recall@0.5 mean = 0.000000

ผล test หลัง calibration และใช้ threshold=0.173797:

- ROC-AUC = 0.500000
- AP = 0.173789
- Sensitivity = 1.000000
- Specificity = 0.000000
- Precision = 0.173789
- F1 = 0.296117
- Accuracy = 0.173789
- Brier = 0.143586
- Log loss = 0.461844

Confusion matrix ใช้ลำดับแถว true 0/1 และคอลัมน์ predicted 0/1:

|  | predicted 0 | predicted 1 |
| --- | --- | --- |
| true 0 | 0 | 870 |
| true 1 | 0 | 183 |

ผลแต่ละ fold อยู่ด้านล่างเพื่อให้เห็นความผันผวนและตรวจค่าเฉลี่ยเองได้

| fold | n fit | n validation | AUC | AP | Recall@0.5 | F1@0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2524 | 632 | 0.500000 | 0.174051 | 0.000000 | 0.000000 |
| 2 | 2525 | 631 | 0.500000 | 0.172742 | 0.000000 | 0.000000 |
| 3 | 2525 | 631 | 0.500000 | 0.172742 | 0.000000 | 0.000000 |
| 4 | 2525 | 631 | 0.500000 | 0.172742 | 0.000000 | 0.000000 |
| 5 | 2525 | 631 | 0.500000 | 0.174326 | 0.000000 | 0.000000 |

### วิเคราะห์ `LogisticRegression` รายตัว

กลไก: เรียน linear relationship บน log-odds พร้อม L2 regularization

พารามิเตอร์หลัก: C=0.5, balanced class weights, max_iter=3000

ข้อควรเข้าใจ: ตีความ coefficient ต้องคำนึงถึง RobustScaler และ features ที่สัมพันธ์กัน; ใช้ sigmoid calibration ครอบอีกชั้น

ผล CV ก่อน calibration:

- ROC-AUC mean = 0.801330
- ROC-AUC SD = 0.017800
- Average Precision mean = 0.447070
- Recall@0.5 mean = 0.756814

ผล test หลัง calibration และใช้ threshold=0.138589:

- ROC-AUC = 0.808649
- AP = 0.400618
- Sensitivity = 0.841530
- Specificity = 0.643678
- Precision = 0.331897
- F1 = 0.476043
- Accuracy = 0.678063
- Brier = 0.120996
- Log loss = 0.373668

Confusion matrix ใช้ลำดับแถว true 0/1 และคอลัมน์ predicted 0/1:

|  | predicted 0 | predicted 1 |
| --- | --- | --- |
| true 0 | 560 | 310 |
| true 1 | 29 | 154 |

ผลแต่ละ fold อยู่ด้านล่างเพื่อให้เห็นความผันผวนและตรวจค่าเฉลี่ยเองได้

| fold | n fit | n validation | AUC | AP | Recall@0.5 | F1@0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2524 | 632 | 0.805399 | 0.427943 | 0.763636 | 0.495575 |
| 2 | 2525 | 631 | 0.787462 | 0.448587 | 0.715596 | 0.464286 |
| 3 | 2525 | 631 | 0.830750 | 0.515077 | 0.770642 | 0.504505 |
| 4 | 2525 | 631 | 0.791047 | 0.426498 | 0.761468 | 0.462396 |
| 5 | 2525 | 631 | 0.791991 | 0.417247 | 0.772727 | 0.467033 |

### วิเคราะห์ `RandomForest` รายตัว

กลไก: bagging ของ decision trees พร้อมสุ่ม features

พารามิเตอร์หลัก: 250 trees, max_depth=8, min_samples_split=10, min_samples_leaf=5, balanced

ข้อควรเข้าใจ: เป็น champion ตาม CV; มี nonlinear interactions แต่ correlated features ทำให้ importance กระจาย

ผล CV ก่อน calibration:

- ROC-AUC mean = 0.807193
- ROC-AUC SD = 0.019175
- Average Precision mean = 0.460996
- Recall@0.5 mean = 0.707706

ผล test หลัง calibration และใช้ threshold=0.123925:

- ROC-AUC = 0.816117
- AP = 0.448405
- Sensitivity = 0.857923
- Specificity = 0.601149
- Precision = 0.311508
- F1 = 0.457060
- Accuracy = 0.645774
- Brier = 0.116224
- Log loss = 0.365408

Confusion matrix ใช้ลำดับแถว true 0/1 และคอลัมน์ predicted 0/1:

|  | predicted 0 | predicted 1 |
| --- | --- | --- |
| true 0 | 523 | 347 |
| true 1 | 26 | 157 |

ผลแต่ละ fold อยู่ด้านล่างเพื่อให้เห็นความผันผวนและตรวจค่าเฉลี่ยเองได้

| fold | n fit | n validation | AUC | AP | Recall@0.5 | F1@0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2524 | 632 | 0.806844 | 0.425524 | 0.636364 | 0.469799 |
| 2 | 2525 | 631 | 0.811593 | 0.462794 | 0.733945 | 0.479042 |
| 3 | 2525 | 631 | 0.835214 | 0.491086 | 0.733945 | 0.509554 |
| 4 | 2525 | 631 | 0.799870 | 0.476155 | 0.770642 | 0.492669 |
| 5 | 2525 | 631 | 0.782446 | 0.449420 | 0.663636 | 0.430678 |

### วิเคราะห์ `ExtraTrees` รายตัว

กลไก: ensemble ของต้นไม้ที่สุ่ม split thresholds มากขึ้น

พารามิเตอร์หลัก: 250 trees, max_depth=8, min_samples_split=10, min_samples_leaf=5, balanced

ข้อควรเข้าใจ: ความสุ่มอาจลด variance แต่ไม่รับประกันว่าจะชนะ Random Forest

ผล CV ก่อน calibration:

- ROC-AUC mean = 0.803254
- ROC-AUC SD = 0.017087
- Average Precision mean = 0.460267
- Recall@0.5 mean = 0.729408

ผล test หลัง calibration และใช้ threshold=0.133950:

- ROC-AUC = 0.802977
- AP = 0.417686
- Sensitivity = 0.808743
- Specificity = 0.635632
- Precision = 0.318280
- F1 = 0.456790
- Accuracy = 0.665717
- Brier = 0.120215
- Log loss = 0.373546

Confusion matrix ใช้ลำดับแถว true 0/1 และคอลัมน์ predicted 0/1:

|  | predicted 0 | predicted 1 |
| --- | --- | --- |
| true 0 | 553 | 317 |
| true 1 | 35 | 148 |

ผลแต่ละ fold อยู่ด้านล่างเพื่อให้เห็นความผันผวนและตรวจค่าเฉลี่ยเองได้

| fold | n fit | n validation | AUC | AP | Recall@0.5 | F1@0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2524 | 632 | 0.803030 | 0.434093 | 0.700000 | 0.468085 |
| 2 | 2525 | 631 | 0.798235 | 0.456755 | 0.733945 | 0.470588 |
| 3 | 2525 | 631 | 0.831242 | 0.500591 | 0.752294 | 0.506173 |
| 4 | 2525 | 631 | 0.784790 | 0.464747 | 0.688073 | 0.432277 |
| 5 | 2525 | 631 | 0.798971 | 0.445150 | 0.772727 | 0.482955 |

### วิเคราะห์ `GradientBoosting` รายตัว

กลไก: เพิ่มต้นไม้แบบ sequential เพื่อแก้ residual/errors

พารามิเตอร์หลัก: 180 stages, learning_rate=0.04, max_depth=4, subsample=0.85

ข้อควรเข้าใจ: ไม่มี class_weight ใน candidate นี้; recall@0.5 จึงเปรียบเทียบกับ balanced raw models อย่างผิวเผินไม่ได้

ผล CV ก่อน calibration:

- ROC-AUC mean = 0.803749
- ROC-AUC SD = 0.014772
- Average Precision mean = 0.455510
- Recall@0.5 mean = 0.212027

ผล test หลัง calibration และใช้ threshold=0.142102:

- ROC-AUC = 0.818359
- AP = 0.454132
- Sensitivity = 0.863388
- Specificity = 0.625287
- Precision = 0.326446
- F1 = 0.473763
- Accuracy = 0.666667
- Brier = 0.116008
- Log loss = 0.365537

Confusion matrix ใช้ลำดับแถว true 0/1 และคอลัมน์ predicted 0/1:

|  | predicted 0 | predicted 1 |
| --- | --- | --- |
| true 0 | 544 | 326 |
| true 1 | 25 | 158 |

ผลแต่ละ fold อยู่ด้านล่างเพื่อให้เห็นความผันผวนและตรวจค่าเฉลี่ยเองได้

| fold | n fit | n validation | AUC | AP | Recall@0.5 | F1@0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2524 | 632 | 0.800331 | 0.424608 | 0.190909 | 0.278146 |
| 2 | 2525 | 631 | 0.805810 | 0.468094 | 0.220183 | 0.309677 |
| 3 | 2525 | 631 | 0.827533 | 0.471724 | 0.110092 | 0.184615 |
| 4 | 2525 | 631 | 0.796988 | 0.472871 | 0.284404 | 0.375758 |
| 5 | 2525 | 631 | 0.788082 | 0.440253 | 0.254545 | 0.345679 |

### วิเคราะห์ `HistGradientBoosting` รายตัว

กลไก: boosting โดยจัด numerical features เป็น bins

พารามิเตอร์หลัก: max_iter=180, learning_rate=0.04, max_depth=5, balanced

ข้อควรเข้าใจ: เป็น fixed configuration; ไม่ใช่ผลที่ผ่าน tuning exhaustive search

ผล CV ก่อน calibration:

- ROC-AUC mean = 0.797670
- ROC-AUC SD = 0.015154
- Average Precision mean = 0.449065
- Recall@0.5 mean = 0.643753

ผล test หลัง calibration และใช้ threshold=0.147892:

- ROC-AUC = 0.805587
- AP = 0.433633
- Sensitivity = 0.830601
- Specificity = 0.618391
- Precision = 0.314050
- F1 = 0.455772
- Accuracy = 0.655271
- Brier = 0.118538
- Log loss = 0.371006

Confusion matrix ใช้ลำดับแถว true 0/1 และคอลัมน์ predicted 0/1:

|  | predicted 0 | predicted 1 |
| --- | --- | --- |
| true 0 | 538 | 332 |
| true 1 | 31 | 152 |

ผลแต่ละ fold อยู่ด้านล่างเพื่อให้เห็นความผันผวนและตรวจค่าเฉลี่ยเองได้

| fold | n fit | n validation | AUC | AP | Recall@0.5 | F1@0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2524 | 632 | 0.808116 | 0.425925 | 0.563636 | 0.441281 |
| 2 | 2525 | 631 | 0.796865 | 0.449503 | 0.669725 | 0.478689 |
| 3 | 2525 | 631 | 0.813350 | 0.458840 | 0.669725 | 0.501718 |
| 4 | 2525 | 631 | 0.796021 | 0.485096 | 0.724771 | 0.484663 |
| 5 | 2525 | 631 | 0.774001 | 0.425959 | 0.590909 | 0.429043 |

### ตีความผล champion เป็นจำนวนคน

Test มี 1053 คน
ในนี้ proxy positive 183 คน และ proxy negative 870 คน
ตรวจพบ positive ได้ 157 คน แต่พลาด 26 คน
ใน proxy negative ทายลบถูก 523 คน และทายบวกผิด 347 คน
Precision=31.15% หมายถึง positive predictions ส่วนใหญ่ยังไม่ใช่ proxy positive
นี่เป็นข้อแลกเปลี่ยนของการเลือก threshold ที่เน้น sensitivity

Accuracy ของ DummyPrior ที่ threshold 0.5 คือ 82.62%
อาจสูงกว่า accuracy ของ screening operating point ทั้งที่ไม่มีความสามารถจัดอันดับเลย
จึงไม่ควรเลือกโมเดลด้วย accuracy เพียงค่าเดียวเมื่อ class imbalance และต้นทุน FN/FP ต่างกัน

## ความไม่แน่นอนและผลแยกกลุ่ม

### ช่วงความไม่แน่นอนของ final test

ใช้ stratified bootstrap 1,000 รอบ seed=42
สุ่มด้วย replacement แยกใน true class 0 และ true class 1
คงจำนวนแต่ละ class เท่ากับ observed test ทุกครั้ง
คำนวณ AUC/AP/sensitivity/specificity ในแต่ละรอบ
ใช้ quantiles 2.5% และ 97.5% เป็น percentile interval

| โมเดล | AUC 95% interval | AP 95% interval | Sensitivity 95% interval | Specificity 95% interval |
| --- | --- | --- | --- | --- |
| DummyPrior | [0.5000, 0.5000] | [0.1738, 0.1738] | [1.0000, 1.0000] | [0.0000, 0.0000] |
| LogisticRegression | [0.7789, 0.8352] | [0.3546, 0.4654] | [0.7869, 0.8907] | [0.6103, 0.6759] |
| RandomForest | [0.7849, 0.8433] | [0.3980, 0.5206] | [0.8087, 0.9071] | [0.5690, 0.6322] |
| ExtraTrees | [0.7719, 0.8303] | [0.3694, 0.4853] | [0.7486, 0.8581] | [0.6023, 0.6655] |
| GradientBoosting | [0.7881, 0.8456] | [0.4032, 0.5259] | [0.8142, 0.9071] | [0.5920, 0.6563] |
| HistGradientBoosting | [0.7756, 0.8334] | [0.3856, 0.5032] | [0.7760, 0.8798] | [0.5851, 0.6494] |

intervals นี้ conditional on fitted model และ observed class counts
ไม่ได้รวมความผันผวนจากการ train ใหม่ การเลือกโมเดล หรือการประมาณ prevalence ใหม่
ไม่ได้ปรับ survey weights, strata หรือ PSU clustering
จึงไม่ใช่ design-based NHANES confidence interval
ไม่ได้ใช้เพื่อประกาศความเหนือกว่าระหว่างโมเดลอย่างเป็นทางการ
การเห็น intervals ทับกันไม่ใช่ formal hypothesis test และการไม่ทับกันก็ต้องดูวิธีเปรียบเทียบให้ถูก

### ตรวจผลแยกกลุ่มของ champion

ตารางนี้เป็น descriptive diagnostic หลังล็อกโมเดลแล้ว
ใช้ threshold เดิมทุกกลุ่ม ไม่ได้ปรับ threshold ให้กลุ่มใดหลังดูผล
ไม่ใช่ fairness certification และจำนวน positive บางกลุ่มน้อย

| กลุ่ม | n | positive | AUC | AP | Sensitivity | Specificity |
| --- | --- | --- | --- | --- | --- | --- |
| female | 564 | 104 | 0.8163 | 0.4644 | 0.8750 | 0.5652 |
| male | 489 | 79 | 0.8135 | 0.4362 | 0.8354 | 0.6415 |
| age_20_39 | 331 | 11 | 0.8233 | 0.1621 | 0.3636 | 0.9500 |
| age_40_59 | 356 | 59 | 0.7882 | 0.4750 | 0.8136 | 0.5354 |
| age_60_plus | 366 | 113 | 0.6850 | 0.4637 | 0.9292 | 0.2372 |

จุดที่ควรอ่านเป็นพิเศษคือกลุ่มอายุ 20–39 มี positive เพียง 11 คน
พบได้ 4 คน พลาด 7 คน sensitivity=36.36%
แม้ AUC รวมจะดูพอใช้ ก็ไม่ควรกล่าวว่า screening นี้เชื่อถือได้เท่ากันทุกช่วงอายุ
กลุ่มอายุ 60+ มี specificity ต่ำ แสดงว่า global threshold ทำให้ false positive มากในกลุ่มนี้
ไม่แก้ด้วยการดู test แล้วตั้ง threshold รายกลุ่มใหม่ทันที
การปรับต้องเป็น protocol ใหม่บน development data และมี evaluation แยกที่เหมาะสม

## SHAP และความหมายของคะแนน

### ทำไมเปลี่ยนจาก TreeExplainer

โมเดลที่ให้บริการจริงมีทั้ง preprocessor, base classifier และ sigmoid calibration
TreeExplainer ของ base classifier เพียงตัวเดียวไม่ได้อธิบาย calibrated output ทั้งชุด
หาก champion เป็น Logistic Regression การเรียก TreeExplainer แบบเดิมก็ไม่รองรับ
ใช้ PermutationExplainer ครอบฟังก์ชันที่คืน calibrated positive-class probability
จึงรองรับ candidate ทุกชนิดผ่าน interface เดียวกัน
ดู [SHAP PermutationExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html)

### Background และจำนวนการประมาณ

สุ่ม background จาก fit partition 32 คน ด้วย seed=42
ใช้ sample จาก fit อีก 100 คน seed=43 เพื่อคำนวณ mean absolute SHAP global importance
max_evals=2×17+1=35 เป็นหนึ่ง forward/reverse permutation ตาม implementation ของ SHAP
individual attribution จึงเป็น approximation ที่อาจเปลี่ยนเมื่อเพิ่ม permutations หรือเปลี่ยน background
แม้แต่ละ contribution มี approximation error ผลรวมยังถูกตรวจ additivity กับ output จริง
ไม่อ้างว่าอันดับ top features มีความเสถียรทางสถิติจากการรันครั้งเดียว

### หน่วยที่ถูกต้อง

probability อยู่ในสเกล 0–1
SHAP contribution ใน space นี้มีหน่วย probability เช่น +0.02
คูณ 100 จะเป็น +2 percentage points หรือ +2 pp
ไม่ใช่ +2% relative risk และไม่ใช่สัดส่วนความสำคัญเทียบกับ abs(SHAP) ทั้งหมด

```python
p = calibrated_model.predict_proba(X)[:, 1]
explanation = explainer(X.to_numpy(), max_evals=35, silent=True)
reconstructed = explanation.base_values + explanation.values.sum(axis=1)
np.testing.assert_allclose(reconstructed, p, atol=1e-7)
# example: base 0.15 + contributions 0.10 = probability 0.25
# display: base 15 pp + total contribution 10 pp = score 25%
```

API ตรวจ sum(base + contributions) ให้ตรงกับ probability ก่อนตอบ
คืน shap_base_score และ shap_total_contribution_points ไว้ตรวจย้อนกลับ
หน้าจอแสดงเฉพาะ top 3 ที่เพิ่ม score และ top 2 ที่ลด score
จึงไม่ควรบวกเฉพาะ 5 รายการบนหน้าจอแล้วคาดว่าจะได้ score เต็ม
ไม่มี factor ติดลบก็คืนรายการว่าง ไม่สร้างคำแนะนำปลอมเติมเข้าไป
field direction ใช้ legacy convention: negative หมายถึงเพิ่ม score และ positive หมายถึงลด score
ควรดูเครื่องหมายของ impact_value ประกอบเพราะชื่อ direction เดิมอาจชวนสับสน

### Global importance ที่คำนวณจริง

| อันดับ | feature | mean(abs(SHAP)) probability units | เทียบเป็น pp |
| --- | --- | --- | --- |
| 1 | age | 0.053065 | 5.3065 |
| 2 | waist_to_height_ratio | 0.036601 | 3.6601 |
| 3 | waist_cm | 0.029587 | 2.9587 |
| 4 | hypertension_flag | 0.023702 | 2.3702 |
| 5 | daily_sugar_g | 0.016251 | 1.6251 |
| 6 | systolic_bp | 0.015022 | 1.5022 |
| 7 | sugar_to_fiber_ratio | 0.008181 | 0.8181 |
| 8 | diastolic_bp | 0.007645 | 0.7645 |
| 9 | bmi | 0.007420 | 0.7420 |
| 10 | daily_calories_kcal | 0.006102 | 0.6102 |
| 11 | weight_kg | 0.005827 | 0.5827 |
| 12 | height_cm | 0.003727 | 0.3727 |
| 13 | daily_fiber_g | 0.003017 | 0.3017 |
| 14 | no_vigorous_recreation_flag | 0.002870 | 0.2870 |
| 15 | gender_is_female | 0.002780 | 0.2780 |
| 16 | sleep_trouble_flag | 0.001927 | 0.1927 |
| 17 | smoking_history_flag | 0.000840 | 0.0840 |

### ข้อจำกัดของ feature attribution

SHAP ใช้ background แทนค่าคอลัมน์เพื่อวัด contribution
features ในโปรเจคมีความสัมพันธ์กันและบางตัวคำนวณจากตัวอื่น
การแทนค่าทีละ feature อาจสร้าง combinations ที่ไม่เกิดจริง เช่น BMI ไม่ตรงน้ำหนัก/ส่วนสูง
global/local importance จึงเป็นคำอธิบายฟังก์ชันของโมเดลภายใต้ masker นี้
ไม่ใช่ causal effect ของการเปลี่ยนปัจจัยนั้น
ไม่ควรบวก contribution ของ waist กับ waist/height แล้วอ้างว่าเป็นผลทางชีววิทยาสองผลแยกกัน
คำว่า score-decreasing association แทน protective factor ใน UI เพื่อลดการตีความเกินหลักฐาน

## Backend และการทำนาย

### ลำดับการทำงานของ POST /api/predict

1. Pydantic ตรวจ schema, numeric bounds และ finite numbers
2. อายุขั้นต่ำ 20 ปีตรงกับ adult cohort
3. ต้องระบุ gender ไม่ตั้ง male ให้เองหากลืมส่ง
4. optional fields ที่ไม่ได้ส่งคงเป็น missing
5. แปลง request เป็น raw-like DataFrame ที่ใช้ชื่อคอลัมน์เดียวกับ training
6. เรียก engineer_features ฟังก์ชันเดียวกับ training
7. จัดลำดับคอลัมน์ด้วย FEATURE_COLUMNS
8. ส่งเข้า calibrated predictor ซึ่งมี fitted preprocessing อยู่ข้างใน
9. อ่าน probability ของ class 1
10. คำนวณ screening_positive จาก full precision probability ไม่ใช่ score ที่ปัดเศษ
11. เลือก display band แยกจาก screening threshold
12. คำนวณ percentile จาก OOF reference ของ fit เท่านั้น
13. คำนวณ SHAP และตรวจ additivity
14. คืน measured BMI/waist ratio, รายการ imputed fields และ score associations

API ไม่เติม BP=120/80 เองเมื่อผู้ใช้ไม่ระบุ
API ไม่คำนวณ median ใหม่จาก request หรือ test
gender coding มีข้อจำกัดตาม binary sex coding ของต้นทาง survey
อายุ 80+ map เป็น 80 เพื่อให้ตรง top-coded training representation
ไม่ได้อ้างว่าสามารถจัดอันดับความเสี่ยงตามอายุจริงในกลุ่มสูงอายุ 80–120 ได้

### Endpoints

| method | path | หน้าที่ |
| --- | --- | --- |
| GET | /health หรือ /api/health | ตรวจ server และการโหลดโมเดล |
| GET | /api/metadata | metrics, threshold, feature stats, importance, limitations |
| POST | /api/predict | calibrated score และ local associations |
| POST | /api/simulate | เปลี่ยน input แล้วคำนวณ score ของ scenario |

### ตัวอย่าง request/response ที่เรียก API จริง

ตัวอย่างด้านล่างเก็บจาก TestClient ผ่าน route ของ FastAPI จริง
score เป็นผลจาก artifact ที่สร้างใน audit นี้
SHAP ราย feature อาจเปลี่ยนเล็กน้อยเมื่อเรียกใหม่เพราะใช้ permutation approximation
probability ของ input เดิมยังคงเดิม

#### `predict` — `/api/predict`

Request:

```json
{
  "age": 55,
  "gender": "male",
  "weight_kg": 90,
  "height_cm": 172,
  "waist_cm": 104,
  "systolic_bp": 138,
  "diastolic_bp": 86,
  "daily_sugar_g": 95,
  "daily_fiber_g": 12,
  "hypertension_history": true,
  "smoking_history": false,
  "vigorous_activity": false,
  "sleep_trouble": true
}
```

Response ที่ได้จริง:

```json
{
  "risk_score": 31.1,
  "risk_category": "Intermediate model score",
  "risk_tier": "moderate",
  "percentile": 79.8,
  "key_drivers": [
    {
      "feature": "Age (80+ top-coded)",
      "raw_feature_name": "age",
      "user_value": "55.00",
      "impact": "+5.71 pp",
      "impact_value": 0.05707319031471701,
      "direction": "negative",
      "advice": "This feature raises the model score relative to its training background. This is a model association, not a causal effect or clinical recommendation."
    },
    {
      "feature": "Waist-to-Height Ratio",
      "raw_feature_name": "waist_to_height_ratio",
      "user_value": "0.60",
      "impact": "+3.23 pp",
      "impact_value": 0.03228736991376416,
      "direction": "negative",
      "advice": "This feature raises the model score relative to its training background. This is a model association, not a causal effect or clinical recommendation."
    },
    {
      "feature": "Hypertension History / Elevated BP",
      "raw_feature_name": "hypertension_flag",
      "user_value": "1.00",
      "impact": "+3.17 pp",
      "impact_value": 0.03165175112334857,
      "direction": "negative",
      "advice": "This feature raises the model score relative to its training background. This is a model association, not a causal effect or clinical recommendation."
    }
  ],
  "protective_factors": [
    {
      "feature": "Mean Diastolic BP",
      "raw_feature_name": "diastolic_bp",
      "user_value": "86.00",
      "impact": "-4.02 pp",
      "impact_value": -0.04020037828934102,
      "direction": "positive",
      "advice": "This feature lowers the model score relative to its training background. This is a model association, not a causal effect or clinical recommendation."
    },
    {
      "feature": "Dietary Fiber",
      "raw_feature_name": "daily_fiber_g",
      "user_value": "12.00",
      "impact": "-0.36 pp",
      "impact_value": -0.003603598369351743,
      "direction": "positive",
      "advice": "This feature lowers the model score relative to its training background. This is a model association, not a causal effect or clinical recommendation."
    }
  ],
  "bmi": 30.4,
  "waist_to_height_ratio": 0.6,
  "imputed_fields": [
    "daily_calories_kcal"
  ],
  "screening_positive": true,
  "decision_threshold": 12.392,
  "shap_base_score": 15.679487,
  "shap_total_contribution_points": 15.397564
}
```

#### `simulate` — `/api/simulate`

Request:

```json
{
  "baseline": {
    "age": 55,
    "gender": "male",
    "weight_kg": 90,
    "height_cm": 172,
    "waist_cm": 104,
    "systolic_bp": 138,
    "diastolic_bp": 86,
    "daily_sugar_g": 95,
    "daily_fiber_g": 12,
    "hypertension_history": true,
    "smoking_history": false,
    "vigorous_activity": false,
    "sleep_trouble": true
  },
  "adjustments": {
    "delta_weight_kg": -5,
    "delta_waist_cm": -4,
    "delta_sugar_pct": -30,
    "delta_fiber_g": 8,
    "active_exercise": true,
    "manage_hypertension": true
  }
}
```

Response ที่ได้จริง:

```json
{
  "original_risk_score": 31.1,
  "new_risk_score": 30.7,
  "delta_risk_score": 0.4,
  "delta_percentage": 1.3,
  "original_category": "Intermediate model score",
  "new_category": "Intermediate model score",
  "new_risk_tier": "moderate",
  "summary_takeaway": "The model score changes from 31.1% to 30.7% (baseline minus scenario: +0.4 percentage points). This is an association-based input scenario, not a forecast of treatment benefit. Missing adjusted measurements use fit-partition medians; values are bounded to the form limits.",
  "simulated_bmi": 28.7,
  "simulated_waist_to_height_ratio": 0.58
}
```

#### `identity_simulation` — `/api/simulate`

Request:

```json
{
  "baseline": {
    "age": 55,
    "gender": "male",
    "weight_kg": 90,
    "height_cm": 172,
    "waist_cm": 104,
    "systolic_bp": 138,
    "diastolic_bp": 86,
    "daily_sugar_g": 95,
    "daily_fiber_g": 12,
    "hypertension_history": true,
    "smoking_history": false,
    "vigorous_activity": false,
    "sleep_trouble": true
  },
  "adjustments": {}
}
```

Response ที่ได้จริง:

```json
{
  "original_risk_score": 31.1,
  "new_risk_score": 31.1,
  "delta_risk_score": 0.0,
  "delta_percentage": 0.0,
  "original_category": "Intermediate model score",
  "new_category": "Intermediate model score",
  "new_risk_tier": "moderate",
  "summary_takeaway": "The model score changes from 31.1% to 31.1% (baseline minus scenario: +0.0 percentage points). This is an association-based input scenario, not a forecast of treatment benefit. Missing adjusted measurements use fit-partition medians; values are bounded to the form limits.",
  "simulated_bmi": 30.4,
  "simulated_waist_to_height_ratio": 0.6
}
```

## What-if และ frontend

### What-if เปลี่ยนอะไรจริง

น้ำหนักเปลี่ยนแล้วคำนวณ BMI ใหม่
รอบเอวเปลี่ยนแล้วคำนวณ waist/height ใหม่
sugar/fiber เปลี่ยนแล้วคำนวณ ratio ใหม่ด้วยกฎเดียวกับ training
activity toggle หมายถึง vigorous recreation ตาม PAQ650
manage_hypertension เปลี่ยน measured BP ลงได้ถึง 120/80 แต่ไม่ลบ history
การเปลี่ยน waist ไม่ได้เดาความสัมพันธ์ใหม่กับ weight โดยอัตโนมัติ
ผู้ใช้สามารถสร้าง combination ที่ไม่น่าเป็นไปได้ทางสรีรวิทยา จึงเป็นเพียง input scenario

หาก baseline ของ feature ที่จะปรับไม่มีค่า ใช้ median ของ fit เป็นฐานของ scenario นั้น
ข้อความ response ระบุพฤติกรรมนี้
ค่าที่ปรับถูก bound ให้ไม่หลุดช่วงที่ schema แบบฟอร์มรับได้
ค่าศูนย์ของ sugar/fiber ไม่ถูกยกขึ้นเป็น 5 g อัตโนมัติเหมือนโค้ดเดิม
adjustments ว่างต้องได้ score เดิมและ delta=0

### หน่วย delta

`delta_risk_score = original_score - new_score`
ค่าบวกหมายถึง model score ต่ำลงกี่ percentage points
`delta_percentage = delta_risk_score / original_score * 100`
เป็น relative change ของ model score ไม่ใช่การลดอุบัติการณ์โรค
ถ้า original_score=0 ให้ relative change เป็น 0 เพื่อหลีกเลี่ยงหารศูนย์
ตัวเลขคำนวณจาก score ที่ปัดหนึ่งตำแหน่ง จึงมี rounding error เล็กน้อยได้

ตัวอย่าง 40% →30% คือ 10 pp หรือ relative decrease 25%
ไม่ใช่ 10% relative decrease
และยังไม่ใช่ข้อพิสูจน์ว่ามาตรการนั้นทำให้ความเสี่ยงจริงลดลง
โมเดล cross-sectional ที่มี known diabetes อยู่ด้วยอาจให้ทิศทางสวนกับ intuition ได้
จึงไม่ควรบังคับ slider ทุกตัวให้ score ลดลงด้วยสูตรเพิ่มเติมที่ไม่มีหลักฐาน

### สิ่งที่แก้บนหน้าเว็บ

- ข้อความนำระบุ educational cross-sectional proxy model
- เอา prediabetes alert และคำแนะนำเชิงสาเหตุจาก SHAP ออก
- แสดง threshold ของ screening แยกจากช่วงสี 20/50%
- percentile ระบุว่าเป็น unweighted training OOF reference
- แสดงรายการ fields ที่ impute
- waist ratio ที่ไม่มี measured waist แสดงว่า missing
- BMI ต่ำกว่า 18.5 ไม่ถูกเรียกว่า normal อัตโนมัติ
- เพิ่มช่อง calories ที่ผู้ใช้แก้ไขได้และเว้นว่างได้ แทนค่า 2,100 kcal ที่เคยส่งแบบซ่อนอยู่
- activity wording ตรง PAQ650
- simulator เริ่มจาก baseline ไม่เปิด intervention โดยอัตโนมัติ
- reset คืน vigorous_activity ตาม baseline
- ใช้ request counter ป้องกัน response ของ slider เก่าทับค่าล่าสุด
- แสดง simulation errors และสถานะ updating
- หน่วย absolute score change เป็น pp

### การเชื่อมต่อ frontend

`frontend/src/lib/api.ts` ใช้ fetch เรียก backend
ค่า default API origin คือ http://localhost:8000
override ผ่าน NEXT_PUBLIC_API_URL ได้ตอนรัน/สร้าง frontend
`page.tsx` เก็บ assessment ล่าสุดและผล prediction
ส่ง baseline เดียวกันต่อให้ WhatIfSimulator
RiskGauge และ RiskDrivers เป็นองค์ประกอบแสดงผล
RecommendationCard ตอนนี้อธิบายวิธีอ่าน score และข้อจำกัดแทน clinical advice ที่สร้างจากเครื่องหมาย SHAP

## วิธีรันตั้งแต่ต้น

### Environment ที่ใช้จริง

| package | version ที่ใช้ |
| --- | --- |
| python | 3.12.13 |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| shap | 0.52.0 |
| joblib | 1.5.3 |
| fastapi | 0.141.1 |

frontend build ตรวจพบ Next.js 14.2.35 ตาม package-lock ที่มีอยู่
README ระบุ version ที่ใช้จริง ไม่ได้หมายความว่าทุก version อื่นจะรันไม่ได้
สำหรับ reproducibility ของ joblib ควรใช้ Python และ library versions ตรงกัน
ไฟล์ requirements-lock.txt เก็บ exact versions ของ Python dependencies ที่ใช้
ตัด LightGBM/XGBoost ที่ติดตั้งเดิมแต่ไม่ได้ใช้ออกจาก lock เพื่อไม่บังคับติดตั้ง runtime ที่ไม่เกี่ยวข้อง

### 1. สร้าง environment และติดตั้ง

```bash
cd "/path/to/CPE310 AI"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
# หากต้องการช่วงเวอร์ชันกว้างแทน exact reproduction:
# python -m pip install -r requirements-dev.txt
```

ต้องมี scikit-learn>=1.6 เพราะใช้ FrozenEstimator
ไม่ต้องใช้ GPU, AWS, external database หรือ API key
ไม่ต้องดาวน์โหลด dataset ใหม่เพราะมี CSV ทั้ง 6 ไฟล์อยู่แล้ว
เมื่อ install ต่างเวอร์ชันควร retrain/re-export แทนการเชื่อว่า joblib เก่า portable เสมอ

### 2. เตรียมข้อมูลและดู cohort

```bash
python -m ml_pipeline.preprocess
```

คำสั่งนี้อ่าน CSV, merge, label และ engineer features
พิมพ์ data audit กับจำนวน missing
ยังไม่ train model และยังไม่แก้ไข raw CSV

### 3. แบ่งข้อมูลและ train/CV

```bash
python -u -m ml_pipeline.train
```

สคริปต์สร้าง fit/calibration/test partitions
รัน 5-fold CV ของ candidate ทั้ง 6
fit final base model ของแต่ละ candidate บน fit partition
บันทึก split, OOF predictions, base models และ manifest
SEQN ของทุก partition อยู่ใน run_manifest.json เพื่อตรวจ patient overlap

### 4. สร้าง EDA ของ fit

```bash
python -m ml_pipeline.eda
```

ต้อง train/split ก่อนจึงค่อยเรียก eda.py
ต่างจากคำสั่งใน README เดิมที่เรียก EDA ก่อนมี split
ตัวเลข EDA ใหม่จะเป็น fit-only ตาม protocol ที่แก้ไข

### 5. Calibrate, evaluate และ export

```bash
python -u -m ml_pipeline.evaluate
```

อ่าน fitted models ที่ล็อกไว้
เรียน calibrator บน calibration partition
เลือก thresholds จาก fit OOF mapped scores
ประเมิน test ของทุก candidate พร้อม bootstrap intervals
สร้าง SHAP artifact สำหรับ champion และตรวจ additivity
export serving model และ metadata ชุดเดียวกัน
หากข้ามขั้นตอนนี้ backend จะไม่มี artifact schema 2 ที่สอดคล้องกับโค้ดใหม่
ต้อง restart backend หลัง re-export เพื่อไม่ให้ process ใช้โมเดลที่โหลดค้างอยู่

### 6. สร้าง audit comparisons และ API examples

```bash
python -m ml_pipeline.audit_report
```

ขั้นตอนนี้ใช้ original source ที่สำรองไว้เพื่อเทียบ label
ไม่ได้ train หรือเลือกโมเดลใหม่
เรียก routes ผ่าน TestClient และบันทึก JSON request/response จริง
subgroup metrics เป็นการตรวจเชิงอธิบายหลังล็อกโมเดล

### 7. ตรวจ regression tests

```bash
python -m unittest discover -s tests -v
```

### 8. สร้าง README จากผลล่าสุด

```bash
python -m ml_pipeline.document
```

คำสั่งนี้เขียน README.md ทับจาก artifacts และ source ปัจจุบัน
อย่ารันก่อนมี metadata/manifest/EDA/audit examples ครบ
หากเปลี่ยน source ให้รันทดสอบส่วนที่เกี่ยวข้องแล้วสร้าง README ใหม่
ไม่ต้องแก้ตารางผลโมเดลด้วยมือทีละค่า

### 9. เปิด backend

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

เปิด Swagger UI ที่ http://localhost:8000/docs
ตรวจ health ที่ http://localhost:8000/health
ใช้ 127.0.0.1 สำหรับการทดลองในเครื่อง
configuration ปัจจุบันเป็น local-development configuration ไม่ได้ผ่าน production deployment review

### 10. เปิด frontend ใน terminal อีกแท็บ

```bash
cd "/path/to/CPE310 AI/frontend"
npm ci
npm run dev
```

เปิด http://localhost:3000
กรอกข้อมูลจริงที่ทราบ ปล่อย optional measurements ว่างเมื่อไม่ทราบ
ตรวจรายการ imputed fields เมื่ออ่านผล
อย่าใช้ผลจาก preset เป็นหลักฐานว่าผู้ใช้จริงต้องอยู่ใน risk tier เดียวกัน

### 11. ตรวจ TypeScript และ production build

```bash
cd "/path/to/CPE310 AI/frontend"
./node_modules/.bin/tsc --noEmit
npm run build
```

## การทดสอบและตรวจซ้ำ

### ผลการตรวจที่รันจริง

- `python -m unittest discover -s tests -v`: ผ่าน 12 tests
- `tsc --noEmit`: ผ่าน
- `npm run build`: ผ่าน production compilation, type checks และ static page generation
- โหลด artifact และเรียก predict/simulate ผ่าน FastAPI TestClient ได้จริง
- ทวน test metrics จาก CSV predictions ได้ตรงกับ metadata
- ทวน SHAP additivity กับ deployed probability ผ่าน

มี deprecation warnings จาก FastAPI/Starlette test client และ joblib/NumPy ใน environment ที่ใช้
ไม่ใช่ test failure แต่ควรตรวจ compatibility อีกครั้งก่อนอัปเกรด dependency
ไม่ได้ทำ manual visual QA ครบทุก browser/device หรือ deployment ภายนอก
production build ผ่านไม่ได้แปลว่า medical validation ผ่าน

### Regression tests ตรวจอะไร

| test | ความผิดพลาดที่ป้องกัน |
| --- | --- |
| target rule | ไม่ใช้ serum glucose>=126 เป็น positive; missing A1c ไม่เป็น control; therapy/diagnosis override |
| medication parser | ไม่จับ glucosamine หรือยาที่ indication กล่าวถึง diabetes อย่างเดียว |
| hypertension unknown | code 7/9 ไม่กลายเป็น no history |
| CV fit boundary | ตรวจจำนวนแถว fit ต่อ fold และ median ที่เรียนจาก fold-training |
| threshold/confusion matrix | ตรวจ max-specificity subject to recall rule บน toy case ที่รู้คำตอบ |
| subject partitions | SEQN ไม่ทับกันและ serving statistics เป็น fit-only |
| train/serve parity | นำแถวจริงมาแปลงผ่าน API แล้วได้ features เดียวกับ training |
| SHAP output | sum ตรวจย้อน probability; signs/units ตรงและ imputed calorie เปิดเผย |
| missing inputs | ไม่มีรอบเอวหรือ BP ก็ไม่สร้าง measured value ปลอม |
| zero fiber/validation | ratio missing ได้, age<20 ถูกปฏิเสธ, gender ต้องส่ง |
| simulation identity/history | ไม่มี adjustments ต้อง delta=0; BP scenario ไม่ลบ diagnosis history |
| exported metrics | คำนวณ test metrics ทุกโมเดลใหม่จาก CSV และตรวจ champion rule |

### วิธีคำนวณ test metrics ซ้ำด้วยตนเอง

```python
import json
import pandas as pd
from ml_pipeline.evaluate import classification_metrics

meta = json.load(open('ml_pipeline/artifacts/metadata.json'))
rows = pd.read_csv('ml_pipeline/artifacts/test_predictions.csv')
for name, frame in rows.groupby('model'):
    threshold = meta['test_benchmarks'][name]['threshold']
    result = classification_metrics(frame.y_true, frame.probability, threshold)
    print(name, result['roc_auc'], result['confusion_matrix'])
```

### วิธีตรวจว่า test ไม่ได้กำหนด champion

```python
import json
meta = json.load(open('ml_pipeline/artifacts/metadata.json'))
best_cv = max(
    (n for n in meta['cv_benchmarks'] if n != 'DummyPrior'),
    key=lambda n: meta['cv_benchmarks'][n]['roc_auc_mean'],
)
assert best_cv == meta['model_name']
# อย่าเปลี่ยนเป็น max(test_benchmarks, key=...test ROC-AUC...)
```

### artifacts แต่ละไฟล์ใช้ตรวจอะไร

| artifact | รายละเอียด |
| --- | --- |
| run_manifest.json | seed, split SEQNs, source hashes, counts, versions, selection rule |
| train_test_splits.joblib | X/y สาม partitions, fit-only stats, OOF, fitted base models |
| benchmark_models.joblib | calibrated candidate models ทุกตัวที่ใช้รายงาน test |
| model_pipeline.joblib | calibrated champion ที่ API เรียกจริง |
| shap_explainer.joblib | explainer, feature names, base value, probability output space |
| metadata.json | CV/test metrics, thresholds, calibration notes, limitations, serving statistics |
| test_predictions.csv | หนึ่งแถวต่อคนต่อโมเดล; y_true, p, raw_p, threshold, predicted class |
| champion_oof_predictions.csv | OOF probability ที่ใช้ threshold/reference ของ champion |
| evaluation_curves.json | ROC, PR และ quantile calibration curve arrays ทุกโมเดล |
| eda_summary.json | fit-only correlations, group means, missingness, statistics |

joblib เป็น Python serialization format ที่มีโค้ด/objects อยู่ได้
โหลดเฉพาะ artifacts ที่สร้างและเชื่อถือได้จากโปรเจคนี้
`audit/original/` เก็บไฟล์ก่อนแก้เพื่อการตรวจย้อนหลัง ไม่ใช่ artifact ที่ backend ใหม่ควรเลือกใช้
ไม่มี Git repository ใน workspace ตอนตรวจ จึงไม่มี commit hash หรือ PR สำหรับการเปลี่ยนครั้งนี้
ใช้ source backups, logs, manifests และ hashes เป็นหลักฐานแทน

## ข้อจำกัดและแนวทางพัฒนา

### ข้อจำกัดที่แก้ด้วยโค้ดชุดนี้อย่างเดียวไม่ได้

1. Cross-sectional data แยกเหตุจากผลไม่ได้
2. Known diabetes ทำให้ lifestyle inputs สะท้อนการรักษาหรือพฤติกรรมหลังวินิจฉัยได้
3. HbA1c ครั้งเดียวไม่ใช่ clinical confirmation ครบกระบวนการ
4. Medication-only label อาจมี other indications
5. ไม่มี reference fasting glucose ใน CSV ที่ใช้สร้าง label ใหม่
6. ไม่ใช้ OGTT เพื่อยืนยันทุกคน และไม่ได้ตรวจ diabetes subtype
7. Control ต้องมี observed HbA1c จึงอาจเกิด selection bias จากผู้ที่เข้ารับการตรวจ/มีข้อมูลครบ
8. ข้อมูล dietary recall วันเดียวไม่ใช่ long-term usual intake
9. Input เว็บเป็น self-report แต่ training มี measured exam values จึงมี measurement shift
10. ไม่ถ่วงน้ำหนัก survey และไม่ได้ทำ design-aware inference
11. random participant split ไม่รับประกัน independence ระดับ household/cluster/geography
12. ข้อมูลสหรัฐฯ 2013–2014 ไม่แทนประชากรไทยปัจจุบัน
13. ยังมี missing data ซึ่ง simple imputation ไม่ถ่ายทอด uncertainty ครบ
14. Global threshold มี performance แตกต่างตามอายุและกลุ่มย่อย
15. SHAP approximation และ correlated/derived features จำกัดการตีความ
16. ชุด test เดิมเคยถูกตรวจมาก่อน การประเมินใหม่จึงต้องการ independent external validation เสริม

### ประเด็น survey design

NHANES ใช้ complex sampling และมี unequal selection probabilities
weights ใช้เมื่อจะประมาณค่าระดับประชากร
ดู [CDC NHANES Weighting Tutorial](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx)
แม้ CSV มี WTMEC2YR/SDMVSTRA/SDMVPSU แต่การเลือก weight ต้องสอดคล้องกับ components และ eligibility
ไม่ใช่ใส่ weight ใดก็ได้แล้วอ้างว่าทุก metric เป็น national estimate ถูกต้อง
ฉบับนี้จึงระบุเป็น unweighted internal sample benchmark และเปลี่ยน population claims ใน UI
หากต้องการ population inference ต้องกำหนด survey analysis protocol เพิ่ม
ควรพิจารณา dietary weights, examination/lab eligibility, strata และ clustered variance ด้วย

### ลำดับพัฒนาต่อที่มีเหตุผล

เริ่มจากยืนยัน intended use ว่าจะจำแนก current diabetes proxy หรือทำนาย incident disease
หากต้องการ future risk ต้องมี longitudinal outcome และ follow-up horizon
หากต้องการ undiagnosed screening ต้องแยก known diagnosis/therapy ออกจาก intended cohort อย่างมี protocol
กำหนด target ด้วย codebook และ reference assays ที่เหมาะสมก่อน optimize model

จากนั้นสร้าง external/temporal validation ที่ไม่ได้ใช้ตัดสินใจพัฒนา
กำหนด clinical costs ของ FN/FP และ threshold protocol ล่วงหน้า
ตรวจ subgroup performance พร้อม uncertainty ที่สอดคล้อง survey design
หากทำ hyperparameter search ให้ทำภายใน development data และใช้ nested CV เมื่อต้องการประเมิน search procedure
ตรวจ calibration/measurement shift ในกลุ่มผู้ใช้จริงและดู missingness pattern

การเพิ่ม XGBoost, deep learning หรือ features มากขึ้นไม่แก้ label error และ leakage โดยอัตโนมัติ
การทำให้ score ลดลงตาม slider ทุกกรณีด้วยสูตรที่เขียนเพิ่มจะทำให้ผลดูน่าเชื่อแต่ไม่ใช่ผลโมเดลจริง
ควรแยก associative score exploration ออกจาก causal intervention evaluation ตั้งแต่ชื่อและข้อความ UI

## โค้ดฉบับเต็มและคำอธิบายรายไฟล์

ส่วนนี้ใส่ source ที่ใช้ในโปรเจคจริงเพื่อให้ตามกระบวนการได้โดยไม่ต้องสลับไฟล์ตลอดเวลา
เป็น snapshot ตอนสร้าง README ด้วย `python -m ml_pipeline.document`
หากแก้ source ภายหลังให้สร้างเอกสารใหม่เพื่อไม่ให้ snippet ล้าสมัย
ไม่รวม binary artifacts และ CSV ดิบทั้งไฟล์เพราะไม่ได้ช่วยอธิบายอัลกอริทึม

### `ml_pipeline/preprocess.py` — การอ่านข้อมูล นิยาม cohort/label และ features

- `read_table`: ตรวจ schema และ primary key ก่อน merge
- `load_and_merge_data`: filter adults/known pregnancy แล้ว left join แบบ one-to-one
- `get_diabetic_medication_seqns`: aggregate ingredient evidence จากหลายแถวต่อคน
- `define_target`: positive/negative/indeterminate โดยไม่ impute outcome
- `safe_sugar_fiber_ratio`: ทำให้ zero-denominator เป็น missing แทนค่ามหาศาล
- `hypertension_indicator`: ไม่ให้ unknown questionnaire กลายเป็น false
- `engineer_features`: สูตรที่ train และ inference ใช้ร่วมกัน
- `describe_features`: summary เท่านั้น; train.py เลือกส่งเฉพาะ fit partition เข้าไปสำหรับ serving
- `build_preprocessor`: median/mode/RobustScaler ที่จะ fit ภายในแต่ละ fold
- `get_prepared_data`: คืน X/y ที่ index เป็น SEQN และข้อมูล audit

```python
"""Deterministic NHANES 2013–2014 cohort, labels and shared feature engineering.
No fitted statistics are learned here; imputation/scaling belong inside each fold.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'ml_pipeline' / 'artifacts'
NUMERICAL_FEATURES = [
    'age', 'bmi', 'waist_cm', 'height_cm', 'weight_kg', 'waist_to_height_ratio',
    'systolic_bp', 'diastolic_bp', 'daily_sugar_g', 'daily_fiber_g',
    'daily_calories_kcal', 'sugar_to_fiber_ratio',
]
BINARY_FEATURES = ['gender_is_female', 'hypertension_flag', 'no_vigorous_recreation_flag',
                   'smoking_history_flag', 'sleep_trouble_flag']
FEATURE_COLUMNS = NUMERICAL_FEATURES + BINARY_FEATURES
# Match whole ingredient tokens, never indications or vague GLIP/GLUCO substrings.
DRUG_PATTERN = (r'\b(?:INSULIN(?: [A-Z]+)?|METFORMIN|GLIPIZIDE|GLYBURIDE|GLIBENCLAMIDE|'
                r'GLIMEPIRIDE|PIOGLITAZONE|ROSIGLITAZONE|SITAGLIPTIN|SAXAGLIPTIN|'
                r'LINAGLIPTIN|ALOGLIPTIN|REPAGLINIDE|NATEGLINIDE|ACARBOSE|MIGLITOL|'
                r'EXENATIDE|LIRAGLUTIDE|CANAGLIFLOZIN|DAPAGLIFLOZIN|EMPAGLIFLOZIN|'
                r'PRAMLINTIDE|TOLBUTAMIDE|TOLAZAMIDE|CHLORPROPAMIDE)\b')


def find_dataset_dir(base_dir=None):
    base = Path(base_dir) if base_dir else ROOT
    for path in (base / 'Dataset', base / 'dataset', base.parent / 'Dataset'):
        if (path / 'demographic.csv').exists():
            return str(path)
    raise FileNotFoundError('Cannot locate Dataset/demographic.csv')


def read_table(dataset_dir, name, columns=None, unique=True):
    path = Path(dataset_dir) / f'{name}.csv'
    # Medication names contain Latin-1 characters; decoding errors must not be discarded.
    df = pd.read_csv(path, encoding='latin1', low_memory=False,
                     usecols=(lambda c: c in columns) if columns else None)
    if columns and (missing := set(columns) - set(df.columns)):
        raise ValueError(f'{name}: missing required columns {sorted(missing)}')
    if df.SEQN.isna().any() or (unique and df.SEQN.duplicated().any()):
        raise ValueError(f'{name}: SEQN must be non-null and unique')
    return df


def get_diabetic_medication_seqns(dataset_dir):
    med = read_table(dataset_dir, 'medications', ['SEQN', 'RXDUSE', 'RXDDRUG'], unique=False)
    mask = med.RXDUSE.eq(1) & med.RXDDRUG.fillna('').str.upper().str.contains(DRUG_PATTERN, regex=True)
    return set(med.loc[mask, 'SEQN'])


def load_and_merge_data(dataset_dir=None):
    dataset_dir = dataset_dir or find_dataset_dir()
    demo = read_table(dataset_dir, 'demographic',
                      ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'SDDSRVYR', 'RIDEXPRG', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA'])
    if not demo.SDDSRVYR.eq(8).all():
        raise ValueError('This pipeline is defined for NHANES 2013–2014 (cycle 8) only')
    adults = demo.loc[demo.RIDAGEYR.ge(20)].copy()
    audit = {'demographic_rows': len(demo), 'adults_before_exclusions': len(adults),
             'known_pregnant_excluded': int(adults.RIDEXPRG.eq(1).sum()), 'tables': {}}
    # Nonpregnant target thresholds; unknown pregnancy remains a documented limitation.
    merged = adults.loc[~adults.RIDEXPRG.eq(1)].copy()
    specs = {
        'examination': ['BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BPXSY1', 'BPXDI1',
                        'BPXSY2', 'BPXDI2', 'BPXSY3', 'BPXDI3'],
        'labs': ['LBXGH', 'LBXSGL'],
        'diet': ['DR1TKCAL', 'DR1TSUGR', 'DR1TFIBE', 'DR1DRSTZ'],
        'questionnaire': ['DIQ010', 'DIQ050', 'DIQ070', 'BPQ020', 'SMQ020', 'PAQ650', 'SLQ050'],
    }
    for name, cols in specs.items():
        table = read_table(dataset_dir, name, ['SEQN'] + cols)
        audit['tables'][name] = {'rows': len(table), 'matched_adults': int(merged.SEQN.isin(table.SEQN).sum())}
        merged = merged.merge(table, how='left', on='SEQN', validate='one_to_one')
    merged['on_diabetic_med'] = merged.SEQN.isin(get_diabetic_medication_seqns(dataset_dir)).astype(int)
    audit['merged_rows'] = len(merged)
    audit['unknown_pregnancy_status_retained'] = int(merged.RIDEXPRG.isna().sum() + merged.RIDEXPRG.eq(3).sum())
    audit['source_files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                             for p in sorted(Path(dataset_dir).glob('*.csv'))}
    merged.attrs['audit'] = audit
    return merged


def define_target(df):
    """Research proxy: HbA1c/diagnosis/therapy; never treat LBXSGL as fasting glucose.
    Controls require an observed HbA1c <6.5 and explicit DIQ010=2.
    HbA1c 5.7–6.4 remains negative for THIS binary diabetes proxy, not 'healthy'.
    Medication-only positives are a proxy and may include other indications.
    """
    df = df.copy()
    a1c = df.LBXGH.where(df.LBXGH.gt(0))
    evidence = pd.DataFrame({
        'a1c_positive': a1c.ge(6.5), 'diagnosis_positive': df.DIQ010.eq(1),
        'insulin_positive': df.DIQ050.eq(1), 'pills_positive': df.DIQ070.eq(1),
        'medication_positive': df.on_diabetic_med.eq(1),
    })
    positive = evidence.any(axis=1)
    negative = df.DIQ010.eq(2) & a1c.notna() & a1c.lt(6.5) & ~positive
    target = pd.Series(np.nan, index=df.index)
    target.loc[negative] = 0
    target.loc[positive] = 1
    df['is_diabetic'] = target
    audit = df.attrs.get('audit', {}).copy()
    old_positive = df.LBXGH.ge(6.5) | df.LBXSGL.ge(126) | df.DIQ010.eq(1) | df.on_diabetic_med.eq(1)
    audit.update({'label_evidence_counts_overlapping': evidence.sum().astype(int).to_dict(),
                  'lbxs_gl_126_without_corrected_positive': int((df.LBXSGL.ge(126) & ~positive).sum()),
                  'old_glucose_rule_positives_with_new_medication_parser': int(old_positive.sum()),
                  'indeterminate_excluded': int(target.isna().sum()),
                  'negative_with_a1c_5_7_to_6_4': int((negative & a1c.ge(5.7)).sum())})
    df = df.loc[target.notna()].copy()
    df['is_diabetic'] = df.is_diabetic.astype(int)
    audit.update({'labeled_rows': len(df), 'positive': int(df.is_diabetic.sum()),
                  'negative': int(df.is_diabetic.eq(0).sum())})
    df.attrs['audit'] = audit
    return df


def safe_sugar_fiber_ratio(sugar, fiber):
    """No ratio when fiber is zero/missing: do not generate values of 60 million."""
    return sugar / fiber.where(fiber.gt(0))


def hypertension_indicator(history, systolic, diastolic):
    """Keep unknown history unknown unless observed BP gives positive evidence."""
    flag = history.map({1: 1.0, 2: 0.0})
    return flag.mask(systolic.ge(140) | diastolic.ge(90), 1.0)


def engineer_features(df):
    df = df.copy()
    df['age'] = df.RIDAGEYR.clip(upper=80)  # 80 is top-coded, not exact age.
    df['gender_is_female'] = df.RIAGENDR.map({1: 0.0, 2: 1.0})
    for raw, feat in [('BMXWT', 'weight_kg'), ('BMXHT', 'height_cm'), ('BMXWAIST', 'waist_cm')]:
        df[feat] = df[raw].where(df[raw].gt(0))
    # Same exact formula for training and web input (not rounded BMI from file).
    df['bmi'] = df.weight_kg / (df.height_cm / 100).pow(2)
    df['waist_to_height_ratio'] = df.waist_cm / df.height_cm
    for prefix, feat in [('BPXSY', 'systolic_bp'), ('BPXDI', 'diastolic_bp')]:
        readings = df[[f'{prefix}{i}' for i in (1, 2, 3)]]
        # Use available positive readings. Zero diastolic is not imputed as normal BP.
        df[feat] = readings.where(readings.gt(0)).mean(axis=1)
    reliable = df.DR1DRSTZ.eq(1)
    for raw, feat in [('DR1TSUGR', 'daily_sugar_g'), ('DR1TFIBE', 'daily_fiber_g'), ('DR1TKCAL', 'daily_calories_kcal')]:
        df[feat] = df[raw].where(reliable & df[raw].ge(0))
    df['sugar_to_fiber_ratio'] = safe_sugar_fiber_ratio(df.daily_sugar_g, df.daily_fiber_g)
    df['hypertension_flag'] = hypertension_indicator(df.BPQ020, df.systolic_bp, df.diastolic_bp)
    df['no_vigorous_recreation_flag'] = df.PAQ650.map({1: 0.0, 2: 1.0})
    df['smoking_history_flag'] = df.SMQ020.map({1: 1.0, 2: 0.0})
    df['sleep_trouble_flag'] = df.SLQ050.map({1: 1.0, 2: 0.0})
    return df


def describe_features(X):
    return {c: {'median': float(X[c].median()), 'mean': float(X[c].mean()),
                'min': float(X[c].min()), 'max': float(X[c].max()),
                'p25': float(X[c].quantile(.25)), 'p75': float(X[c].quantile(.75)),
                'missing_n': int(X[c].isna().sum())} for c in FEATURE_COLUMNS}


def build_preprocessor():
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scaler', RobustScaler())]), NUMERICAL_FEATURES),
        ('bin', SimpleImputer(strategy='most_frequent', keep_empty_features=True), BINARY_FEATURES),
    ], remainder='drop')


def get_prepared_data(dataset_dir=None):
    df = engineer_features(define_target(load_and_merge_data(dataset_dir)))
    X = df.set_index('SEQN')[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = df.set_index('SEQN').is_diabetic
    X.attrs['audit'] = df.attrs['audit']
    return X, y, describe_features(X)


if __name__ == '__main__':
    X, y, _ = get_prepared_data()
    print(X.attrs['audit'])
    print(X.isna().sum())
```

### `ml_pipeline/train.py` — การแบ่งข้อมูล ทำ CV และเลือก base model

- `get_candidate_models`: candidate configurations ที่กำหนดตายตัว
- `build_model`: clone classifier และสร้าง preprocessor ใหม่ทุกครั้ง
- `response_scores`: จัดรูป response ให้ตรงวิธี calibrate ของ sklearn
- `benchmark_models`: fit/validate ทีละ fold, เก็บ OOF และ fold metrics
- `train_and_export`: split 60/20/20, ตรวจ overlap, fit-only stats และ manifest

```python
"""Fit-only CV selection; independent calibration and final test partitions."""
import json
import platform
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score
from ml_pipeline.preprocess import ARTIFACTS, build_preprocessor, get_prepared_data, describe_features

SEED = 42


def get_candidate_models():
    # Fixed, declared candidates. This is benchmarking, NOT hyperparameter tuning.
    return {
        'DummyPrior': DummyClassifier(strategy='prior', random_state=SEED),
        'LogisticRegression': LogisticRegression(class_weight='balanced', C=.5, max_iter=3000, random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_split=10,
                          min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state=SEED),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=250, max_depth=8, min_samples_split=10,
                          min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state=SEED),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=180, learning_rate=.04,
                          max_depth=4, subsample=.85, random_state=SEED),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=180, learning_rate=.04,
                          max_depth=5, class_weight='balanced', random_state=SEED),
    }


def build_model(classifier):
    return Pipeline([('preprocessor', build_preprocessor()), ('classifier', clone(classifier))])


def response_scores(model, X):
    # Match sklearn's calibration input: decision_function when available, else probability.
    return model.decision_function(X) if hasattr(model, 'decision_function') else model.predict_proba(X)[:, 1]


def benchmark_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, oof, fitted = {}, {}, {}
    for name, classifier in get_candidate_models().items():
        fold_results = []
        probabilities = np.full(len(y_train), np.nan)
        responses = np.full(len(y_train), np.nan)
        for fold, (tr, va) in enumerate(cv.split(X_train, y_train), 1):
            model = build_model(classifier)
            model.fit(X_train.iloc[tr], y_train.iloc[tr])
            p = model.predict_proba(X_train.iloc[va])[:, 1]
            probabilities[va] = p
            responses[va] = response_scores(model, X_train.iloc[va])
            yy = y_train.iloc[va]
            fold_results.append({'fold': fold, 'n_fit': len(tr), 'n_validation': len(va),
                'roc_auc': float(roc_auc_score(yy, p)),
                'average_precision': float(average_precision_score(yy, p)),
                'recall_at_0_5': float(recall_score(yy, p >= .5, zero_division=0)),
                'f1_at_0_5': float(f1_score(yy, p >= .5, zero_division=0))})
        results[name] = {'folds': fold_results}
        for metric in ('roc_auc', 'average_precision', 'recall_at_0_5', 'f1_at_0_5'):
            values = [f[metric] for f in fold_results]
            results[name][metric + '_mean'] = float(np.mean(values))
            results[name][metric + '_std'] = float(np.std(values, ddof=1))
        oof[name] = {'probabilities': probabilities, 'responses': responses}
        final = build_model(classifier).fit(X_train, y_train)
        fitted[name] = final
        print(f"{name}: CV AUC={results[name]['roc_auc_mean']:.4f}, AP={results[name]['average_precision_mean']:.4f}", flush=True)
    eligible = [n for n in results if n != 'DummyPrior']
    best = max(eligible, key=lambda n: results[n]['roc_auc_mean'])
    return best, results, fitted, oof


def train_and_export(output_dir=ARTIFACTS):
    output_dir = __import__('pathlib').Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, _ = get_prepared_data()
    # Reserve test before fitting or exploring training distributions.
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=SEED)
    ids = {'fit': X_fit.index.tolist(), 'calibration': X_cal.index.tolist(), 'test': X_test.index.tolist()}
    assert not (set(ids['fit']) & set(ids['test']) or set(ids['fit']) & set(ids['calibration']) or set(ids['test']) & set(ids['calibration']))
    best, cv, fitted, oof = benchmark_models(X_fit, y_fit)
    splits = {'X_train': X_fit, 'y_train': y_fit, 'X_cal': X_cal, 'y_cal': y_cal,
              'X_test': X_test, 'y_test': y_test, 'feature_stats': describe_features(X_fit),
              'best_model_name': best, 'cv_results': cv, 'fitted_models': fitted, 'oof': oof}
    joblib.dump(splits, output_dir / 'train_test_splits.joblib')
    manifest = {'seed': SEED, 'selection_rule': 'Highest mean fit-only 5-fold ROC-AUC; test never selects model',
                'partitions': {k: {'n': len(v), 'positive': int(y.loc[v].sum()), 'seqn': v} for k,v in ids.items()},
                'data_audit': X.attrs['audit'],
                'versions': {'python': platform.python_version(), **{p: version(p) for p in ['numpy','pandas','scikit-learn','shap','joblib','fastapi']}}}
    (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Selected {best}; fit={len(X_fit)}, calibration={len(X_cal)}, test={len(X_test)}', flush=True)
    return splits


if __name__ == '__main__':
    train_and_export()
```

### `ml_pipeline/evaluate.py` — Calibration, threshold, test และ export

- `classification_metrics`: confusion matrix และ metrics ทั้ง threshold-dependent/independent
- `select_threshold`: บังคับ sensitivity target ก่อน maximize specificity
- `bootstrap_intervals`: uncertainty แบบ stratified IID conditional bootstrap
- `run_evaluation`: ใช้ champion ที่เลือกไว้แล้ว, freeze base estimator, calibrate และทดสอบ
- ส่วน SHAP ครอบ deployed model จริง ไม่เลือกอธิบายเฉพาะ estimator ภายใน
- test_predictions.csv และ metadata ใช้ค่าชุดเดียวกัน ไม่ hardcode จำนวน test 1,131

```python
"""Independent sigmoid calibration, locked test comparison and probability SHAP."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    recall_score, precision_score, f1_score, confusion_matrix, brier_score_loss,
    log_loss, roc_curve, precision_recall_curve)
from ml_pipeline.preprocess import ARTIFACTS, FEATURE_COLUMNS
from ml_pipeline.explain import ProbabilityPredictor, compute_sample_shap
from ml_pipeline.train import SEED


def classification_metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'threshold': float(threshold), 'roc_auc': float(roc_auc_score(y,p)),
        'average_precision': float(average_precision_score(y,p)),
        'sensitivity': float(recall_score(y,pred,zero_division=0)),
        'specificity': float(tn/(tn+fp)), 'precision': float(precision_score(y,pred,zero_division=0)),
        'f1': float(f1_score(y,pred,zero_division=0)), 'accuracy': float(accuracy_score(y,pred)),
        'brier': float(brier_score_loss(y,p)), 'log_loss': float(log_loss(y,p,labels=[0,1])),
        'confusion_matrix': {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}}


def select_threshold(y, probabilities, sensitivity_target=.8):
    # Highest specificity subject to >= target recall on fit-only OOF predictions.
    fpr, tpr, thresholds = roc_curve(y, probabilities, drop_intermediate=False)
    valid = np.flatnonzero((tpr >= sensitivity_target) & np.isfinite(thresholds))
    i = min(valid, key=lambda k: (fpr[k], -thresholds[k]))
    return float(thresholds[i])


def bootstrap_intervals(y, p, threshold, repeats=1000):
    # Stratified IID bootstrap: conditional on this fitted model and observed sample.
    # Not survey/design-aware and excludes training/selection uncertainty.
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(y == c) for c in (0,1)]
    draws = {k: [] for k in ('roc_auc','average_precision','sensitivity','specificity')}
    for _ in range(repeats):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        yy, pp = y[ix], p[ix]
        draws['roc_auc'].append(roc_auc_score(yy,pp))
        draws['average_precision'].append(average_precision_score(yy,pp))
        draws['sensitivity'].append(float((pp[yy==1] >= threshold).mean()))
        draws['specificity'].append(float((pp[yy==0] < threshold).mean()))
    return {k: [float(v) for v in np.quantile(a,[.025,.975])] for k,a in draws.items()}


def run_evaluation(artifacts_dir=ARTIFACTS):
    out = Path(artifacts_dir)
    splits = joblib.load(out / 'train_test_splits.joblib')
    X_fit,y_fit = splits['X_train'],splits['y_train']
    X_cal,y_cal = splits['X_cal'],splits['y_cal']
    X_test,y_test = splits['X_test'],splits['y_test']
    best = splits['best_model_name']  # Already selected; never rerank using test results.
    benchmarks, calibrated_models, oof_probs, prediction_rows = {}, {}, {}, []
    curves = {}
    for name, raw in splits['fitted_models'].items():
        calibrated = CalibratedClassifierCV(FrozenEstimator(raw), method='sigmoid', ensemble=False)
        calibrated.fit(X_cal,y_cal)
        # sklearn uses decision_function for LR and predict_proba for trees.
        sigmoid = calibrated.calibrated_classifiers_[0].calibrators[0]
        oof = sigmoid.predict(splits['oof'][name]['responses'])
        threshold = select_threshold(y_fit,oof)
        p = calibrated.predict_proba(X_test)[:,1]
        raw_p = raw.predict_proba(X_test)[:,1]
        metrics = classification_metrics(y_test,p,threshold)
        metrics['at_0_5'] = classification_metrics(y_test,p,.5)
        metrics['raw_uncalibrated'] = classification_metrics(y_test,raw_p,.5)
        metrics['ci95_stratified_iid_bootstrap'] = bootstrap_intervals(y_test,p,threshold)
        metrics['threshold_fit_oof'] = classification_metrics(y_fit,oof,threshold)
        benchmarks[name] = metrics
        calibrated_models[name],oof_probs[name] = calibrated,oof
        fpr,tpr,rt=roc_curve(y_test,p)
        precision,recall,pt=precision_recall_curve(y_test,p)
        obs,pred=calibration_curve(y_test,p,n_bins=8,strategy='quantile')
        curves[name] = {'roc': {'fpr': fpr.tolist(),'tpr':tpr.tolist()},
                        'precision_recall': {'precision':precision.tolist(),'recall':recall.tolist()},
                        'calibration': {'mean_predicted':pred.tolist(),'fraction_positive':obs.tolist()}}
        prediction_rows.extend({'SEQN': int(seqn), 'model':name, 'y_true':int(yy),
            'probability':float(pp),'raw_probability':float(rp),'threshold':threshold,'prediction':int(pp>=threshold)}
            for seqn, yy, pp, rp in zip(X_test.index,y_test,p,raw_p))
        print(f'{name}: test AUC={metrics["roc_auc"]:.4f}, AP={metrics["average_precision"]:.4f}, recall={metrics["sensitivity"]:.4f}, threshold={threshold:.4f}',flush=True)
    pipeline = calibrated_models[best]
    background = X_fit.sample(n=min(32,len(X_fit)),random_state=SEED)
    explainer = shap.PermutationExplainer(ProbabilityPredictor(pipeline), background.to_numpy(),
        feature_names=FEATURE_COLUMNS, seed=SEED)
    sample = X_fit.sample(n=min(100,len(X_fit)),random_state=SEED+1)
    explanation = explainer(sample.to_numpy(),max_evals=2*len(FEATURE_COLUMNS)+1,silent=True)
    explained_probability = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(explained_probability,pipeline.predict_proba(sample)[:,1],atol=1e-7)
    importance = sorted([{'feature':c,'importance':float(v)} for c,v in
        zip(FEATURE_COLUMNS,np.abs(explanation.values).mean(axis=0))],key=lambda r:r['importance'],reverse=True)
    ref = oof_probs[best]
    m = benchmarks[best]
    metadata = {'schema_version':2,'model_name':best,
        'task':'Cross-sectional diabetes proxy classification; not incident diabetes or prediabetes diagnosis',
        'metrics':{'test_roc_auc':m['roc_auc'],'test_pr_auc':m['average_precision'],
            'test_recall_sensitivity':m['sensitivity'],'test_specificity':m['specificity'],
            'test_precision':m['precision'],'test_f1':m['f1'],'test_accuracy':m['accuracy'],
            'test_brier':m['brier'],'confusion_matrix':m['confusion_matrix']},
        'cv_benchmarks':splits['cv_results'],'test_benchmarks':benchmarks,
        'feature_names':FEATURE_COLUMNS,'raw_feature_columns':FEATURE_COLUMNS,
        'feature_stats':splits['feature_stats'],'feature_stats_source':'fit partition only',
        'feature_importance':importance,'importance_method':'Permutation SHAP on calibrated probability; 32 fit background rows, 100 fit explanation rows, one forward/reverse permutation',
        'explanation_units':'probability; multiply by 100 for percentage points',
        'percentile_bins':np.quantile(ref,np.linspace(0,1,101)).tolist(),
        'percentile_source':'fit-only out-of-fold scores mapped through independent sigmoid calibration; unweighted sample reference',
        'decision_threshold':m['threshold'],'threshold_rule':'fit OOF sensitivity >=0.80, maximize specificity; no test tuning',
        'risk_thresholds':{'low_max':20.,'moderate_max':50.,'high_min':50.},
        'risk_categories':{'low':'Lower model score','moderate':'Intermediate model score','high':'Higher model score'},
        'risk_tiers_note':'20/50 percent are display bands, not validated diagnostic thresholds',
        'calibration':'Sigmoid fitted on independent 20% calibration partition; base models fitted on 60%; final 20% test',
        'test_prevalence':float(y_test.mean()),'test_n':len(y_test),
        'limitations':['Unweighted 2013–2014 US sample; not population prevalence or validated Thai risk',
            'Single-visit HbA1c/diagnosis/therapy proxy; not clinical confirmation',
            'Known diabetes included; dietary reverse causation and treatment effects possible',
            'Survey clusters and weights not modeled; IID CIs are approximate',
            'SHAP and what-if scores are associations, not intervention effects',
            'One split and preexisting historical test exposure; independent external validation still required']}
    joblib.dump(pipeline,out/'model_pipeline.joblib')
    joblib.dump({'explainer':explainer,'feature_names':FEATURE_COLUMNS,
                'base_value':float(explanation.base_values[0]),'output_space':'calibrated_probability'},out/'shap_explainer.joblib')
    joblib.dump(calibrated_models,out/'benchmark_models.joblib')
    pd.DataFrame(prediction_rows).to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame({'SEQN':X_fit.index,'y_true':y_fit,'calibrated_oof_probability':ref}).to_csv(out/'champion_oof_predictions.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    (out/'evaluation_curves.json').write_text(json.dumps(curves,indent=2,allow_nan=False))
    return metadata


if __name__ == '__main__':
    run_evaluation()
```

### `ml_pipeline/explain.py` — Callable สำหรับอธิบาย calibrated probabilities

- `ProbabilityPredictor`: แปลง NumPy ที่ SHAP ส่งมาเป็น DataFrame ตาม feature order
- `compute_sample_shap`: คืน contributions ตามชื่อ features ของ request หนึ่งแถว
- การ define callable ใน module ทำให้ joblib deserialize ได้เมื่อ backend โหลด

```python
"""Serializable predictor for SHAP on the actual calibrated probability output."""
import numpy as np
import pandas as pd
from ml_pipeline.preprocess import FEATURE_COLUMNS


class ProbabilityPredictor:
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        frame = pd.DataFrame(np.asarray(X), columns=FEATURE_COLUMNS)
        return self.model.predict_proba(frame)[:, 1]


def compute_sample_shap(explainer, feature_names, X):
    values = explainer(np.asarray(X), max_evals=2 * len(feature_names) + 1, silent=True)
    return {name: float(v) for name, v in zip(feature_names, values.values[0])}
```

### `ml_pipeline/eda.py` — EDA หลังแบ่งข้อมูลแล้ว

- อ่าน split artifact แทนการ summarize target ของ test
- คำนวณ correlation, group means และ missingness เฉพาะ fit partition
- ไม่ได้ใช้ผลนี้เลือก feature เพิ่มหลังดู test

```python
"""Fit-partition exploratory summaries; never inspect test labels to select features."""
import json
from pathlib import Path
import joblib
from ml_pipeline.preprocess import ARTIFACTS, describe_features


def run_eda(output_dir=ARTIFACTS):
    out=Path(output_dir)
    split=joblib.load(out/'train_test_splits.joblib')
    X,y=split['X_train'],split['y_train']
    df=X.assign(is_diabetic=y)
    summary={'scope':'fit partition only, unweighted', 'dataset_size':len(X),
        'prevalence':float(y.mean()),'class_distribution':{'diabetic':int(y.sum()),'non_diabetic':int((1-y).sum())},
        'feature_correlations':df.corr().is_diabetic.drop('is_diabetic').sort_values(ascending=False).to_dict(),
        'group_means':df.groupby('is_diabetic').mean().T.rename(columns={0:'proxy_negative',1:'proxy_positive'}).to_dict(),
        'missing_percentages':(X.isna().mean()*100).to_dict(),'feature_stats':describe_features(X)}
    (out/'eda_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    run_eda()
```

### `backend/app/schemas.py` — ข้อตกลง input/output ของ API

- `AssessmentInput`: bounds, optional missing fields, age>=20 และ gender ที่ต้องส่ง
- `FeatureImpact`: SHAP contribution ใน probability units และข้อความ pp
- `PredictionResponse`: score, threshold result, missing fields และ additivity summary
- `SimulationDelta`: ช่วง delta ที่อนุญาตและ nullable activity override
- `SimulationResponse`: คะแนนเดิม/ใหม่และการเปลี่ยน score

```python
"""
Pydantic schemas for request and response validation.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class AssessmentInput(BaseModel):
    """User assessment inputs from non-invasive screening form."""
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    age: float = Field(..., ge=20, le=120, description="Adults 20+; ages 80+ map to NHANES top-coded age 80")
    gender: Literal["male", "female"] = Field(..., description="Biological sex ('male' or 'female')")
    weight_kg: float = Field(..., ge=30, le=250, description="Body weight in kilograms")
    height_cm: float = Field(..., ge=100, le=240, description="Standing height in centimeters")
    waist_cm: Optional[float] = Field(default=None, ge=40, le=180, description="Waist circumference in cm")
    systolic_bp: Optional[float] = Field(default=None, ge=70, le=240, description="Systolic blood pressure (mmHg)")
    diastolic_bp: Optional[float] = Field(default=None, ge=40, le=140, description="Diastolic blood pressure (mmHg)")
    
    # Dietary metrics (daily)
    daily_sugar_g: Optional[float] = Field(default=None, ge=0, le=500, description="Daily total sugar intake in grams")
    daily_fiber_g: Optional[float] = Field(default=None, ge=0, le=150, description="Daily dietary fiber intake in grams")
    daily_calories_kcal: Optional[float] = Field(default=None, ge=500, le=6000, description="Estimated daily caloric intake")
    
    # Questionnaire & Lifestyle indicators
    hypertension_history: Optional[bool] = Field(default=None, description="Told by doctor of high blood pressure")
    smoking_history: Optional[bool] = Field(default=None, description="Smoked 100+ cigarettes in lifetime")
    vigorous_activity: Optional[bool] = Field(default=None, description="Any vigorous recreational activity lasting at least 10 continuous minutes in a typical week (PAQ650)")
    sleep_trouble: Optional[bool] = Field(default=None, description="Ever told a doctor about trouble sleeping (SLQ050)")


class FeatureImpact(BaseModel):
    """Local SHAP contribution to the model score."""
    feature: str = Field(..., description="Human-readable feature name")
    raw_feature_name: str = Field(..., description="Internal feature column key")
    user_value: Any = Field(..., description="User's input or estimated value")
    impact: str = Field(..., description="SHAP contribution in percentage points, e.g. +1.2 pp")
    impact_value: float = Field(..., description="Raw SHAP impact value")
    direction: Literal["negative", "positive"] = Field(..., description="Legacy signs: 'negative' raises model score, 'positive' lowers it")
    advice: str = Field(..., description="Association-only explanation; not a causal or clinical recommendation")


class PredictionResponse(BaseModel):
    """Complete prediction output with explainability."""
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Calibrated cross-sectional diabetes-proxy probability as percentage")
    risk_category: str = Field(..., description="Illustrative score band; not a diagnostic category")
    risk_tier: Literal["low", "moderate", "high"] = Field(..., description="Normalized tier key for UI styling")
    percentile: float = Field(..., ge=0.0, le=100.0, description="Percentile against unweighted fit OOF model scores")
    key_drivers: List[FeatureImpact] = Field(default_factory=list, description="Largest score-increasing SHAP associations")
    protective_factors: List[FeatureImpact] = Field(default_factory=list, description="Largest score-decreasing SHAP associations")
    bmi: float = Field(..., description="Calculated Body Mass Index")
    waist_to_height_ratio: Optional[float] = Field(..., description="Observed waist-to-height ratio, null when waist omitted")
    imputed_fields: List[str] = Field(default_factory=list)
    screening_positive: bool
    decision_threshold: float
    shap_base_score: float
    shap_total_contribution_points: float


class SimulationDelta(BaseModel):
    """User adjustments for lifestyle What-If simulation."""
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    delta_weight_kg: float = Field(default=0.0, ge=-40.0, le=20.0, description="Weight change in kg (e.g. -5 for 5kg loss)")
    delta_waist_cm: float = Field(default=0.0, ge=-30.0, le=20.0, description="Waist reduction in cm (e.g. -4)")
    delta_sugar_pct: float = Field(default=0.0, ge=-90.0, le=100.0, description="Sugar intake percentage change (e.g. -40%)")
    delta_fiber_g: float = Field(default=0.0, ge=-20.0, le=40.0, description="Fiber increase in grams (e.g. +8g)")
    active_exercise: Optional[bool] = Field(default=None, description="Switch to regular vigorous physical activity")
    manage_hypertension: Optional[bool] = Field(default=None, description="Set current BP scenario at most 120/80 while preserving diagnosis history")


class SimulationRequest(BaseModel):
    """Simulation request containing baseline inputs and proposed delta changes."""
    baseline: AssessmentInput
    adjustments: SimulationDelta


class SimulationResponse(BaseModel):
    """Results of the What-If simulation."""
    original_risk_score: float
    new_risk_score: float
    delta_risk_score: float
    delta_percentage: float
    original_category: str
    new_category: str
    new_risk_tier: Literal["low", "moderate", "high"]
    summary_takeaway: str
    simulated_bmi: float
    simulated_waist_to_height_ratio: Optional[float]
```

### `backend/app/model_service.py` — Serving ที่ใช้สูตรเดียวกับ training

- `_input_to_dataframe`: map request กลับ raw-like columns แล้วใช้ engineer_features
- `predict`: calibrated output, categories, OOF percentile, SHAP และ additivity check
- `simulate`: apply deltas, preserve history, clamp bounds และคำนวณ scenario
- ทุก score-decreasing factor มาจาก contribution ติดลบจริง

```python
"""Serve the actual calibrated classifier and association-only SHAP/what-if outputs."""
import json
import joblib
import numpy as np
import pandas as pd
from backend.app.config import settings
from backend.app.schemas import AssessmentInput, FeatureImpact, PredictionResponse, SimulationRequest, SimulationResponse
from ml_pipeline.preprocess import FEATURE_COLUMNS, engineer_features
from ml_pipeline.explain import compute_sample_shap

FEATURE_DISPLAY_NAMES = {
    'age':'Age (80+ top-coded)', 'gender_is_female':'Biological Sex', 'bmi':'BMI',
    'waist_cm':'Waist Circumference', 'height_cm':'Height', 'weight_kg':'Weight',
    'waist_to_height_ratio':'Waist-to-Height Ratio', 'systolic_bp':'Mean Systolic BP',
    'diastolic_bp':'Mean Diastolic BP', 'daily_sugar_g':'Total Sugar (one-day recall)',
    'daily_fiber_g':'Dietary Fiber', 'daily_calories_kcal':'Calories',
    'sugar_to_fiber_ratio':'Sugar-to-Fiber Ratio', 'hypertension_flag':'Hypertension History / Elevated BP',
    'no_vigorous_recreation_flag':'No Vigorous Recreational Activity',
    'smoking_history_flag':'Ever Smoked 100+ Cigarettes', 'sleep_trouble_flag':'Ever Told Doctor of Sleep Trouble',
}


class ModelService:
    def __init__(self):
        self.pipeline = joblib.load(settings.pipeline_path)
        artifact = joblib.load(settings.explainer_path)
        self.explainer = artifact['explainer']
        self.feature_names = artifact['feature_names']
        self.base_value = artifact['base_value']
        with open(settings.metadata_path) as f:
            self.metadata = json.load(f)
        if self.metadata.get('schema_version') != 2:
            raise ValueError('Artifacts must be rebuilt with the audited pipeline (schema version 2)')

    def get_metadata(self):
        return self.metadata

    def _input_to_dataframe(self, inp):
        def code(value):
            return np.nan if value is None else (1 if value else 2)
        row = {'RIDAGEYR':inp.age,'RIAGENDR':2 if inp.gender=='female' else 1,
               'BMXWT':inp.weight_kg,'BMXHT':inp.height_cm,'BMXWAIST':inp.waist_cm,
               'DR1TSUGR':inp.daily_sugar_g,'DR1TFIBE':inp.daily_fiber_g,
               'DR1TKCAL':inp.daily_calories_kcal,'DR1DRSTZ':1,
               'BPQ020':code(inp.hypertension_history),'PAQ650':code(inp.vigorous_activity),
               'SMQ020':code(inp.smoking_history),'SLQ050':code(inp.sleep_trouble)}
        for i in (1,2,3):
            row[f'BPXSY{i}'] = inp.systolic_bp if i==1 else np.nan
            row[f'BPXDI{i}'] = inp.diastolic_bp if i==1 else np.nan
        raw = pd.DataFrame([row]).astype(float)
        return engineer_features(raw)[FEATURE_COLUMNS]

    def _determine_category(self, score_pct):
        if score_pct < 20:
            return 'Lower model score','low'
        if score_pct <= 50:
            return 'Intermediate model score','moderate'
        return 'Higher model score','high'

    def _calculate_percentile(self, prob):
        bins = self.metadata['percentile_bins']
        return round(float(np.interp(prob,bins,np.linspace(0,100,len(bins)))),1)

    def predict(self, inp):
        X = self._input_to_dataframe(inp)
        prob = float(self.pipeline.predict_proba(X)[0,1])
        category,tier = self._determine_category(prob*100)
        contributions = compute_sample_shap(self.explainer,self.feature_names,X)
        # Check the explanation against the deployed calibrated output.
        if not np.isclose(self.base_value+sum(contributions.values()),prob,atol=1e-6):
            raise RuntimeError('SHAP contributions do not reconstruct deployed probability')
        increasing, decreasing = [],[]
        for feat,value in contributions.items():
            raw = X.iloc[0][feat]
            if abs(value) <= 1e-8:
                continue
            direction = 'raises' if value>0 else 'lowers'
            factor = FeatureImpact(feature=FEATURE_DISPLAY_NAMES[feat],raw_feature_name=feat,
                user_value='Missing (imputed by fitted pipeline)' if pd.isna(raw) else f'{raw:.2f}',
                impact=f'{value*100:+.2f} pp',impact_value=value,
                direction='negative' if value>0 else 'positive',
                advice=f'This feature {direction} the model score relative to its training background. This is a model association, not a causal effect or clinical recommendation.')
            (increasing if value>0 else decreasing).append(factor)
        increasing.sort(key=lambda v:abs(v.impact_value),reverse=True)
        decreasing.sort(key=lambda v:abs(v.impact_value),reverse=True)
        whtr = X.waist_to_height_ratio.iloc[0]
        return PredictionResponse(risk_score=round(prob*100,1),risk_category=category,risk_tier=tier,
            percentile=self._calculate_percentile(prob),key_drivers=increasing[:3],protective_factors=decreasing[:2],
            bmi=round(float(X.bmi.iloc[0]),1),waist_to_height_ratio=None if pd.isna(whtr) else round(float(whtr),2),
            imputed_fields=X.columns[X.iloc[0].isna()].tolist(),
            screening_positive=prob>=self.metadata['decision_threshold'],
            decision_threshold=round(self.metadata['decision_threshold']*100,3),
            shap_base_score=round(self.base_value*100,6),
            shap_total_contribution_points=round(sum(contributions.values())*100,6))

    def simulate(self, req):
        baseline,adj = req.baseline,req.adjustments
        original = self.predict(baseline)
        d = baseline.model_dump()
        stats = self.metadata['feature_stats']
        def resolved(field):
            return d[field] if d[field] is not None else stats[field]['median']
        def clamp(value, low, high):
            return float(np.clip(value,low,high))
        if adj.delta_weight_kg:
            d['weight_kg']=clamp(baseline.weight_kg+adj.delta_weight_kg,30,250)
        if adj.delta_waist_cm:
            d['waist_cm']=clamp(resolved('waist_cm')+adj.delta_waist_cm,40,180)
        if adj.delta_sugar_pct:
            d['daily_sugar_g']=clamp(resolved('daily_sugar_g')*(1+adj.delta_sugar_pct/100),0,500)
        if adj.delta_fiber_g:
            d['daily_fiber_g']=clamp(resolved('daily_fiber_g')+adj.delta_fiber_g,0,150)
        if adj.active_exercise is not None:
            d['vigorous_activity']=adj.active_exercise
        if adj.manage_hypertension:
            d['systolic_bp']=min(resolved('systolic_bp'),120)
            d['diastolic_bp']=min(resolved('diastolic_bp'),80)
            # A current BP scenario cannot erase past diagnosis.
        simulated=self.predict(AssessmentInput(**d)) if d!=baseline.model_dump() else original
        delta=round(original.risk_score-simulated.risk_score,1)
        relative=round(delta/original.risk_score*100,1) if original.risk_score else 0.
        summary=(f'The model score changes from {original.risk_score}% to {simulated.risk_score}% '
                 f'(baseline minus scenario: {delta:+.1f} percentage points). '
                 'This is an association-based input scenario, not a forecast of treatment benefit. '
                 'Missing adjusted measurements use fit-partition medians; values are bounded to the form limits.')
        return SimulationResponse(original_risk_score=original.risk_score,new_risk_score=simulated.risk_score,
            delta_risk_score=delta,delta_percentage=relative,original_category=original.risk_category,
            new_category=simulated.risk_category,new_risk_tier=simulated.risk_tier,summary_takeaway=summary,
            simulated_bmi=simulated.bmi,simulated_waist_to_height_ratio=simulated.waist_to_height_ratio)


model_service=ModelService()
```

### `tests/test_pipeline.py` — Tests ที่รันจริงสำหรับ invariants และ integration

- มีทั้งตัวอย่างที่ทราบคำตอบและการตรวจ artifact จริง
- ตรวจ CV fit boundary ผ่าน recording pipeline โดยไม่ต้อง retrain candidate ใหญ่
- ตรวจ API ผ่าน FastAPI routes ไม่ได้เรียกเพียง helper function

```python
"""Scientific/data invariants and deployed-artifact regression checks (unittest)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fastapi.testclient import TestClient
from ml_pipeline.preprocess import (ARTIFACTS, FEATURE_COLUMNS, build_preprocessor, define_target,
    engineer_features, get_diabetic_medication_seqns, hypertension_indicator, get_prepared_data)
from ml_pipeline.evaluate import classification_metrics, select_threshold
from ml_pipeline.train import benchmark_models


class ScientificInvariants(unittest.TestCase):
    def test_target_does_not_use_nonfasting_glucose_or_missing_a1c_as_control(self):
        df=pd.DataFrame({'LBXGH':[5.5,np.nan,6.5,6.1,5.1,5.0],
            'LBXSGL':[190,90,90,90,90,90], 'DIQ010':[2,2,2,2,1,2],
            'DIQ050':[2,2,2,2,2,1], 'DIQ070':[2]*6, 'on_diabetic_med':[0]*6})
        labeled=define_target(df)
        self.assertNotIn(1,labeled.index)
        self.assertEqual(labeled.is_diabetic.tolist(),[0,1,0,1,1])

    def test_medication_match_is_not_an_indication_or_substring(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({'SEQN':[1,2,3,4,5], 'RXDUSE':[1,1,1,1,2],
                'RXDDRUG':['METFORMIN; SITAGLIPTIN','GLUCOSAMINE','OLMESARTAN','INSULIN GLARGINE','METFORMIN'],
                'RXDRSD1':['Other','Other','Type 2 diabetes mellitus','Other','Other']}).to_csv(Path(d)/'medications.csv',index=False)
            self.assertEqual(get_diabetic_medication_seqns(d),{1,4})

    def test_hypertension_unknown_not_treated_as_no(self):
        flag=hypertension_indicator(pd.Series([9.,7.,2.,1.]),pd.Series([120.,150.,120.,120.]),pd.Series([80.]*4))
        self.assertTrue(pd.isna(flag.iloc[0]))
        self.assertEqual(flag.iloc[1:].tolist(),[1.,0.,1.])

    def test_fold_preprocessing_fits_only_training_rows(self):
        rng=np.random.default_rng(10)
        X=pd.DataFrame(rng.normal(size=(50,len(FEATURE_COLUMNS))),columns=FEATURE_COLUMNS)
        X.iloc[::4,1]=np.nan
        y=pd.Series([0,1]*25)
        fits=[]
        class RecordingPipeline(Pipeline):
            def fit(self,X,y,**kwargs):
                super().fit(X,y,**kwargs)
                actual=self.named_steps['preprocessor'].named_transformers_['num'].named_steps['imputer'].statistics_
                np.testing.assert_allclose(actual,np.nanmedian(X.iloc[:,:12],axis=0))
                fits.append(len(X))
                return self
        def model(clf):
            return RecordingPipeline([('preprocessor',build_preprocessor()),('classifier',clf)])
        with patch('ml_pipeline.train.get_candidate_models',return_value={
                'DummyPrior':DummyClassifier(strategy='prior'),'LogisticRegression':LogisticRegression()}), \
             patch('ml_pipeline.train.build_model',side_effect=model):
            benchmark_models(X,y)
        self.assertEqual(fits.count(40),10)
        self.assertEqual(fits.count(50),2)

    def test_threshold_and_confusion_matrix(self):
        y=[0,0,1,1]
        p=[.1,.2,.3,.9]
        t=select_threshold(y,p)
        self.assertEqual(t,.3)
        self.assertEqual(classification_metrics(y,p,t)['confusion_matrix'],{'tn':2,'fp':0,'fn':0,'tp':2})


class ArtifactIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.app.main import app
        from backend.app.model_service import model_service
        cls.client=TestClient(app)
        cls.service=model_service
        cls.splits=joblib.load(ARTIFACTS/'train_test_splits.joblib')
        cls.input={'age':55,'gender':'male','weight_kg':90,'height_cm':172,'waist_cm':104,
                   'systolic_bp':138,'diastolic_bp':86,'daily_sugar_g':95,'daily_fiber_g':12,
                   'hypertension_history':True,'smoking_history':False,'vigorous_activity':True,'sleep_trouble':True}

    def test_disjoint_subjects_and_fit_only_statistics(self):
        s=self.splits
        groups=[set(s[k].index) for k in ['X_train','X_cal','X_test']]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        for col,stats in s['feature_stats'].items():
            self.assertEqual(stats['median'],s['X_train'][col].median())
        self.assertFalse(set(FEATURE_COLUMNS)&{'SEQN','LBXGH','LBXSGL','DIQ010','DIQ050','DIQ070','on_diabetic_med'})

    def test_training_and_api_feature_engineering_are_identical(self):
        from backend.app.schemas import AssessmentInput
        X,y,_=get_prepared_data()
        complete=X.dropna()
        row=complete.iloc[0]
        inp=AssessmentInput(age=row.age,gender='female' if row.gender_is_female else 'male',
            weight_kg=row.weight_kg,height_cm=row.height_cm,waist_cm=row.waist_cm,
            systolic_bp=row.systolic_bp,diastolic_bp=row.diastolic_bp,
            daily_sugar_g=row.daily_sugar_g,daily_fiber_g=row.daily_fiber_g,daily_calories_kcal=row.daily_calories_kcal,
            hypertension_history=bool(row.hypertension_flag),vigorous_activity=not bool(row.no_vigorous_recreation_flag),
            smoking_history=bool(row.smoking_history_flag),sleep_trouble=bool(row.sleep_trouble_flag))
        actual=self.service._input_to_dataframe(inp)
        np.testing.assert_allclose(actual.iloc[0],row,atol=1e-12)

    def test_predict_shap_reconstructs_and_outputs_real_signs(self):
        res=self.client.post('/api/predict',json=self.input)
        self.assertEqual(res.status_code,200,res.text)
        self.assertTrue(self.client.get('/health').json()['model_loaded'])
        self.assertEqual(self.client.get('/api/metadata').json()['schema_version'],2)
        r=res.json()
        self.assertAlmostEqual(r['shap_base_score']+r['shap_total_contribution_points'],r['risk_score'],delta=.051)
        for factor in r['key_drivers']:
            self.assertGreater(factor['impact_value'],0)
            self.assertTrue(factor['impact'].endswith(' pp'))
        for factor in r['protective_factors']:
            self.assertLess(factor['impact_value'],0)
        self.assertEqual(r['imputed_fields'],['daily_calories_kcal'])

    def test_missing_inputs_are_missing_not_invented_measurements(self):
        res=self.client.post('/api/predict',json={'age':90,'gender':'male','height_cm':170,'weight_kg':70})
        self.assertEqual(res.status_code,200,res.text)
        r=res.json()
        self.assertIsNone(r['waist_to_height_ratio'])
        self.assertIn('waist_cm',r['imputed_fields'])
        self.assertIn('systolic_bp',r['imputed_fields'])

    def test_zero_fiber_and_invalid_age(self):
        res=self.client.post('/api/predict',json={**self.input,'daily_fiber_g':0})
        self.assertEqual(res.status_code,200,res.text)
        self.assertIn('sugar_to_fiber_ratio',res.json()['imputed_fields'])
        self.assertEqual(self.client.post('/api/predict',json={**self.input,'age':19}).status_code,422)
        without_gender={k:v for k,v in self.input.items() if k!='gender'}
        self.assertEqual(self.client.post('/api/predict',json=without_gender).status_code,422)

    def test_identity_simulation_and_bp_history(self):
        r=self.client.post('/api/simulate',json={'baseline':self.input,'adjustments':{}})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['delta_risk_score'],0)
        from backend.app.schemas import SimulationRequest
        captured=[]
        original=self.service.predict
        def record(inp):
            captured.append(inp)
            return original(inp)
        with patch.object(self.service,'predict',side_effect=record):
            self.service.simulate(SimulationRequest(baseline=self.input,adjustments={'manage_hypertension':True}))
        self.assertTrue(captured[-1].hypertension_history)
        self.assertEqual(captured[-1].vigorous_activity,True)

    def test_saved_test_predictions_reproduce_metrics(self):
        rows=pd.read_csv(ARTIFACTS/'test_predictions.csv')
        meta=self.service.metadata
        for name,frame in rows.groupby('model'):
            recorded=meta['test_benchmarks'][name]
            actual=classification_metrics(frame.y_true,frame.probability,recorded['threshold'])
            self.assertEqual(actual['confusion_matrix'],recorded['confusion_matrix'])
            self.assertAlmostEqual(actual['roc_auc'],recorded['roc_auc'],places=12)
        self.assertEqual(meta['model_name'],max((n for n in meta['cv_benchmarks'] if n!='DummyPrior'),
            key=lambda n:meta['cv_benchmarks'][n]['roc_auc_mean']))


if __name__=='__main__':
    unittest.main(verbosity=2)
```

### `frontend/src/components/WhatIfSimulator.tsx` — State และ input scenarios ของหน้าเว็บ

- เริ่มต้นด้วย adjustments ศูนย์และ activity เท่ากับ baseline
- request ID ป้องกัน response เก่าทับ response ใหม่เมื่อเลื่อนเร็ว
- แสดง score difference เป็น pp พร้อมข้อความ association-only
- reset คืน state ให้ตรงกับ baseline รวม activity

```tsx
'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { AssessmentFormData, SimulationAdjustments, SimulationResult } from '../types';
import { runSimulation } from '../lib/api';

interface WhatIfSimulatorProps {
  baseline: AssessmentFormData;
  initialScore: number;
}

export default function WhatIfSimulator({ baseline, initialScore }: WhatIfSimulatorProps) {
  const [adjustments, setAdjustments] = useState<SimulationAdjustments>({
    delta_weight_kg: 0,
    delta_waist_cm: 0,
    delta_sugar_pct: 0,
    delta_fiber_g: 0,
    active_exercise: baseline.vigorous_activity,
    manage_hypertension: false,
  });

  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationError, setSimulationError] = useState<string | null>(null);
  const latestRequest = useRef(0);

  const executeSimulation = useCallback(async (currentAdjustments: SimulationAdjustments) => {
    const requestId = ++latestRequest.current;
    setIsSimulating(true);
    setSimulationError(null);
    try {
      const result = await runSimulation(baseline, currentAdjustments);
      if (requestId === latestRequest.current) setSimulation(result);
    } catch (err: any) {
      if (requestId === latestRequest.current) setSimulationError(err.message || 'Simulation failed.');
    } finally {
      if (requestId === latestRequest.current) setIsSimulating(false);
    }
  }, [baseline]);

  useEffect(() => {
    const reset: SimulationAdjustments = {
      delta_weight_kg: 0, delta_waist_cm: 0, delta_sugar_pct: 0, delta_fiber_g: 0,
      active_exercise: baseline.vigorous_activity, manage_hypertension: false,
    };
    setAdjustments(reset);
    executeSimulation(reset);
    return () => { latestRequest.current += 1; };
  }, [baseline, executeSimulation]);

  const handleSliderChange = (field: keyof SimulationAdjustments, value: any) => {
    const nextAdjustments = { ...adjustments, [field]: value };
    setAdjustments(nextAdjustments);
    executeSimulation(nextAdjustments);
  };

  const handleReset = () => {
    const zeroAdjustments: SimulationAdjustments = {
      delta_weight_kg: 0,
      delta_waist_cm: 0,
      delta_sugar_pct: 0,
      delta_fiber_g: 0,
      active_exercise: baseline.vigorous_activity,
      manage_hypertension: false,
    };
    setAdjustments(zeroAdjustments);
    executeSimulation(zeroAdjustments);
  };

  return (
    <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 border-b border-slate-100 pb-2.5">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Model Input Scenario Simulator
        </h3>
        <button
          type="button"
          onClick={handleReset}
          className="text-xs text-slate-500 hover:text-slate-800 underline underline-offset-2 cursor-pointer"
        >
          Reset to Baseline
        </button>
      </div>

      {/* Simulated Outcome Banner (Light Theme) */}
      {simulation && (
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">
              Scenario Model Score
            </span>
            <div className="flex items-baseline gap-2.5 mt-0.5">
              <span className="text-3xl font-extrabold tracking-tight text-slate-900">
                {simulation.new_risk_score.toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500">
                (Baseline: {simulation.original_risk_score.toFixed(1)}%)
              </span>
            </div>
            <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-600">
              <span className="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-700 text-[11px] font-medium">
                {simulation.new_category}
              </span>
              <span>
                BMI: <strong className="text-slate-800">{simulation.simulated_bmi}</strong> | WHtR: <strong className="text-slate-800">{simulation.simulated_waist_to_height_ratio}</strong>
              </span>
            </div>
          </div>

          {simulation.delta_risk_score > 0 ? (
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-3.5 py-2 text-left sm:text-right shrink-0">
              <span className="text-emerald-700 text-xs font-bold block">
                -{simulation.delta_risk_score.toFixed(1)} pp Lower Model Score
              </span>
              <span className="text-[11px] text-emerald-600">
                ({simulation.delta_percentage.toFixed(1)}% relative decrease)
              </span>
            </div>
          ) : simulation.delta_risk_score < 0 ? (
            <div className="bg-rose-50 border border-rose-200 rounded-lg px-3.5 py-2 text-left sm:text-right shrink-0">
              <span className="text-rose-700 text-xs font-bold block">
                +{Math.abs(simulation.delta_risk_score).toFixed(1)} pp Higher Model Score
              </span>
            </div>
          ) : (
            <div className="bg-white border border-slate-200 rounded-lg px-3.5 py-2 text-left sm:text-right shrink-0">
              <span className="text-xs text-slate-600 font-medium block">Baseline Values</span>
              <span className="text-[11px] text-slate-400">Adjust sliders to test changes</span>
            </div>
          )}
        </div>
      )}

      <p className="text-[11px] text-slate-500">{simulation?.summary_takeaway || 'Changes show model associations, not causal treatment outcomes.'}</p>
      {isSimulating && <p className="text-xs text-slate-500">Updating scenario...</p>}
      {simulationError && <p role="alert" className="text-xs text-rose-700">{simulationError}</p>}
      {/* Sliders */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
        {/* Waist */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-600">Waist Trim</span>
            <span className="font-semibold text-slate-800">
              {adjustments.delta_waist_cm < 0 ? `${adjustments.delta_waist_cm} cm` : '0'}
            </span>
          </div>
          <input
            type="range"
            min="-15"
            max="0"
            step="1"
            value={adjustments.delta_waist_cm}
            onChange={(e) => handleSliderChange('delta_waist_cm', Number(e.target.value))}
            className="w-full h-1 bg-slate-300 rounded appearance-none cursor-pointer accent-slate-800"
          />
        </div>

        {/* Weight */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-600">Weight Loss</span>
            <span className="font-semibold text-slate-800">
              {adjustments.delta_weight_kg < 0 ? `${adjustments.delta_weight_kg} kg` : '0'}
            </span>
          </div>
          <input
            type="range"
            min="-15"
            max="0"
            step="1"
            value={adjustments.delta_weight_kg}
            onChange={(e) => handleSliderChange('delta_weight_kg', Number(e.target.value))}
            className="w-full h-1 bg-slate-300 rounded appearance-none cursor-pointer accent-slate-800"
          />
        </div>

        {/* Sugar */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-600">Sugar Cut</span>
            <span className="font-semibold text-slate-800">
              {adjustments.delta_sugar_pct < 0 ? `${adjustments.delta_sugar_pct}%` : '0'}
            </span>
          </div>
          <input
            type="range"
            min="-80"
            max="0"
            step="5"
            value={adjustments.delta_sugar_pct}
            onChange={(e) => handleSliderChange('delta_sugar_pct', Number(e.target.value))}
            className="w-full h-1 bg-slate-300 rounded appearance-none cursor-pointer accent-slate-800"
          />
        </div>

        {/* Fiber */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-600">Fiber Boost</span>
            <span className="font-semibold text-slate-800">
              {adjustments.delta_fiber_g > 0 ? `+${adjustments.delta_fiber_g}g` : '0'}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="25"
            step="1"
            value={adjustments.delta_fiber_g}
            onChange={(e) => handleSliderChange('delta_fiber_g', Number(e.target.value))}
            className="w-full h-1 bg-slate-300 rounded appearance-none cursor-pointer accent-slate-800"
          />
        </div>
      </div>

      {/* Checkboxes */}
      <div className="flex flex-wrap items-center gap-4 pt-1 text-xs">
        <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-700">
          <input
            type="checkbox"
            checked={adjustments.active_exercise}
            onChange={(e) => handleSliderChange('active_exercise', e.target.checked)}
            className="w-3.5 h-3.5 rounded border-slate-300 text-slate-800 focus:ring-0"
          />
          <span>Any vigorous recreation (10+ continuous min)</span>
        </label>

        <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-700">
          <input
            type="checkbox"
            checked={adjustments.manage_hypertension}
            onChange={(e) => handleSliderChange('manage_hypertension', e.target.checked)}
            className="w-3.5 h-3.5 rounded border-slate-300 text-slate-800 focus:ring-0"
          />
          <span>Set measured BP to at most 120/80; retain history</span>
        </label>
      </div>
    </section>
  );
}
```

## แหล่งอ้างอิงหลัก

ลิงก์อ้างอิงใช้สำหรับความหมายตัวแปรและวิธีการ ไม่ใช่เป็นแหล่งตัวเลข performance ของโปรเจค
ตัวเลข performance ทั้งหมดมาจาก artifacts และ logs ที่รันใน workspace นี้

- [CDC DEMO_H: รอบสำรวจและ demographics](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.htm)
- [CDC BIOPRO_H: standard biochemistry glucose](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BIOPRO_H.htm)
- [CDC DIQ_H: diagnosis และ therapy questions](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DIQ_H.htm)
- [CDC PAQ_H: activity questionnaire](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAQ_H.htm)
- [CDC DR1TOT_H: one-day dietary recall](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DR1TOT_H.htm)
- [CDC NHANES weighting](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx)
- [NIDDK: HbA1c และการตรวจยืนยัน](https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test/)
- [sklearn: Pipeline/ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
- [sklearn: calibration ของ fitted models](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- [sklearn: Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [SHAP: PermutationExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html)
