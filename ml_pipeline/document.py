"""Generate the detailed Thai README from actual audit artifacts and current source."""
import json
from pathlib import Path
from ml_pipeline.preprocess import ROOT, ARTIFACTS, FEATURE_COLUMNS


def generate():
    m=json.loads((ARTIFACTS/'metadata.json').read_text())
    manifest=json.loads((ARTIFACTS/'run_manifest.json').read_text())
    eda=json.loads((ARTIFACTS/'eda_summary.json').read_text())
    labels=json.loads((ROOT/'audit/label_comparison.json').read_text())
    examples=json.loads((ROOT/'audit/api_examples.json').read_text())
    subgroups=json.loads((ROOT/'audit/subgroup_metrics.json').read_text())
    a=manifest['data_audit']
    chunks=[]
    def add(s): chunks.append(s.strip()+'\n')
    def code(s,lang='python'): add(f'```{lang}\n{s.rstrip()}\n```')
    def table(headers,rows):
        add('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows))
    add('''# GlycoRisk — รายงานตรวจสอบและคู่มือโปรเจค Data Science ตั้งแต่ข้อมูลดิบจนถึง Web Application

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
''')
    findings=[
        ('CV preprocessing leakage','fit_transform(X_train) หนึ่งครั้งก่อนแบ่ง fold','สร้าง Pipeline ใหม่ภายในแต่ละ fold; imputer/scaler เรียนจาก fold-training เท่านั้น'),
        ('ใช้ glucose ผิดชนิด','เรียก LBXSGL ว่า fasting glucose และตัดที่ 126','เอา LBXSGL ออกจากกฎ label; ใช้ HbA1c/diagnosis/therapy proxy ที่มีหลักฐานในไฟล์'),
        ('control ไม่มี biomarker','ยอมให้ HbA1c และ glucose หายทั้งคู่แล้วยังเป็น negative','negative ต้องมี HbA1c <6.5 ที่สังเกตจริง และ DIQ010=2'),
        ('ค้นหายากว้างเกินไป','ค้น GLUCO/DIABET ในชื่อยาและเหตุผลที่ใช้ยา','จับชื่อสารออกฤทธิ์เต็มคำใน RXDDRUG และ RXDUSE=1; aggregate เป็นรายคน'),
        ('ตาราง benchmark เดิมปะปน','recall/AP บางรายการเป็นค่า CV แต่แสดงเหมือน test','export ผล test ของทุก candidate แยกจากผล CV ชัดเจน'),
        ('ไม่มี baseline ที่ไม่ใช้ features','เปรียบเทียบเฉพาะโมเดล ML','เพิ่ม DummyPrior เพื่อดูค่า AUC/AP/Brier ที่ต้องชนะ'),
        ('class weighting กับ probability','นำ predict_proba ของ balanced classifier แสดงเป็นความเสี่ยงตรง ๆ','ใช้ sigmoid calibration บน partition แยกจาก model fitting'),
        ('threshold 0.5 โดยไม่มีเหตุผล','ใช้ 0.5 กับทุกโมเดลทั้งที่สเกล probability ต่างกัน','เลือก threshold จาก OOF ของชุดฝึก ให้ sensitivity อย่างน้อย 80% และ specificity สูงสุดภายใต้กฎนี้'),
        ('statistics ใช้ทั้ง cohort','ค่า median fallback รวม test เข้าไปด้วย','feature_stats สำหรับ serving คำนวณจาก fit partition เท่านั้น'),
        ('เดารอบเอวเอง','ใช้ส่วนสูงคูณ 0.48 หรือ 0.54 เมื่อไม่มีรอบเอว','ส่ง missing เข้า fitted imputer; ไม่สร้างรอบเอวเสมือนว่าเป็นค่าที่วัดจริง'),
        ('calories ที่ซ่อนใน form','state ส่งค่า 2,100 kcal ทั้งที่ไม่มีช่องให้ผู้ใช้แก้','เพิ่ม optional calories input และให้ค่าเริ่มต้นว่าง'),
        ('train/serve feature ต่างกัน','train ใช้ BMI ที่ปัดเศษในไฟล์ แต่ API คำนวณอีกสูตร','เรียก engineer_features ฟังก์ชันเดียวกันทั้ง training และ API'),
        ('sugar/fiber ระเบิดเมื่อ fiber=0','หารด้วย fiber+1e-5 ทำให้บางค่าเกิน 60 ล้าน','fiber<=0 ทำให้ ratio เป็น missing แล้วให้ fold-specific imputer จัดการ'),
        ('activity ผิดความหมาย','PAQ605 เป็นกิจกรรมหนักจากงาน แต่ UI บอกออกกำลังกาย','ใช้ PAQ650 กิจกรรมหนักเพื่อสันทนาการ; เปลี่ยนข้อความ UI ให้ตรงคำถาม'),
        ('ไม่ทำกิจกรรมหนัก = sedentary','ตั้งชื่อ sedentary_flag จากคำถามกิจกรรมหนักข้อเดียว','ใช้ชื่อ no_vigorous_recreation_flag; ไม่อ้างว่าเป็นการนั่งมากหรือไม่มีกิจกรรมทุกชนิด'),
        ('unknown history กลายเป็น false','BPQ020=7/9 ถูกมองคล้ายไม่มีประวัติ','map เฉพาะ 1/2; unknown คงเป็น missing เว้นแต่ BP ให้ positive evidence'),
        ('ปลอม protective factor','หากไม่มี SHAP ติดลบ ใส่ Dietary Fiber -3.5% เอง','คืนรายการว่างได้; ทุก factor ต้องมาจาก SHAP จริง'),
        ('SHAP ใช้หน่วยผิด','เอาสัดส่วน abs(SHAP) ไปใส่เครื่องหมาย %','อธิบาย calibrated probability โดยตรงและใช้ percentage points หรือ pp'),
        ('ความเข้ากันได้ของ explainer','เดิมรองรับเฉพาะต้นไม้; เมื่อเพิ่ม calibration ต้องอธิบาย wrapper ให้ครบ','ใช้ PermutationExplainer รองรับ champion ทุกชนิดและ calibrated predictor ทั้งชุด'),
        ('pre-diabetes จาก score band','เรียกคะแนน 20–50% ว่า prediabetes alert','ใช้ชื่อช่วงคะแนนทั่วไป; แยกสีจาก threshold คัดกรอง'),
        ('what-if อ้างเหตุและผล','ข้อความสัญญาว่าลดน้ำหนักแล้วจะลดความเสี่ยงจริงเท่านั้นเท่านี้','รายงานการเปลี่ยน score ภายใต้ input scenario; ไม่สรุป treatment effect'),
        ('ลบประวัติความดันใน simulation','จัดการความดันแล้วตั้ง hypertension_history=False','เปลี่ยนเฉพาะค่าความดันปัจจุบัน; ประวัติการวินิจฉัยยังอยู่'),
        ('reset เปลี่ยน activity โดยไม่ตั้งใจ','reset ตั้ง exercise=False แม้ baseline เป็น True','คืน activity ตาม baseline และเริ่ม sliders ที่ศูนย์'),
        ('claim ระดับประชากร','เรียก unweighted training percentile ว่า percentile ประชากร','ใช้ OOF reference ของชุดฝึกและระบุว่า unweighted sample'),
        ('ความไม่แน่นอนไม่ชัด','มีเพียง point estimates','เพิ่ม stratified IID bootstrap 1,000 ครั้ง พร้อมข้อจำกัดเรื่อง survey design'),
    ]
    table(['ประเด็น','เดิม','แก้แล้ว'],findings)
    add('''การแก้ครั้งนี้ไม่ได้ปรับสูตรหรือเลือกโมเดลเพื่อดัน test AUC ให้สูงที่สุด
โมเดลชนะถูกเลือกด้วย CV ของชุด fit ก่อนเปิดใช้ test ใน workflow ใหม่
แม้ GradientBoosting จะมี test AUC สูงกว่าเล็กน้อย ก็ไม่เปลี่ยน champion ตาม test
อย่าเปรียบเทียบ AUC เดิม 0.7790 กับ AUC ใหม่ 0.8161 แล้วสรุปว่าดีขึ้นเป็นเหตุจากการแก้เพียงอย่างเดียว
เพราะ label, cohort, partition และรายละเอียด features เปลี่ยนหลายส่วนพร้อมกัน

## ไฟล์และโครงสร้าง
''')
    code('''CPE310 AI/
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
└── README.md''','text')
    add('''directory จริงใช้ `Dataset` ตัว D ใหญ่
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
''')
    table(['ไฟล์','แถว','คอลัมน์','SEQN ไม่ซ้ำ','บทบาท'],[
        ['demographic.csv',10175,47,10175,'ฐานผู้เข้าร่วม อายุ เพศ รอบสำรวจ design variables'],
        ['examination.csv',9813,224,9813,'น้ำหนัก ส่วนสูง รอบเอว ความดัน'],
        ['labs.csv',9813,424,9813,'HbA1c ใช้สร้าง label; LBXSGL ใช้ audit เท่านั้น'],
        ['diet.csv',9813,168,9813,'total nutrients จาก one-day recall และ quality status'],
        ['questionnaire.csv',10175,953,10175,'diagnosis, therapy และ lifestyle'],
        ['medications.csv',20194,13,10175,'หลายแถวต่อคน ต้อง aggregate ก่อน join'],
    ])
    add('''### ตรวจโครงสร้างก่อนนำข้อมูลเข้าระบบ

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
''')
    table(['ไฟล์','จำนวน bytes','SHA-256'],[(n,v['bytes'],v['sha256']) for n,v in a['source_files'].items()])
    add('''## การรวมตารางและคัดเลือกผู้เข้าร่วม

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
''')
    table(['ขั้นตอน','จำนวนคน'],[
        ['demographic ทั้งหมด',a['demographic_rows']],['อายุ >=20',a['adults_before_exclusions']],
        ['known pregnancy ที่ตัดออก',a['known_pregnant_excluded']],['หลังตัด known pregnancy / หลัง merge',a['merged_rows']],
        ['label ไม่ชัดเจนที่ตัดออก',a['indeterminate_excluded']],['analytic cohort',a['labeled_rows']],
        ['proxy positive',a['positive']],['proxy negative',a['negative']],
    ])
    add('''## นิยาม label ที่แก้ไข

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
''')
    code('''positive = (
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
# positive -> 1; negative -> 0; ที่เหลือ -> missing label และตัดออก''')
    add('''### ยาต้องอ่านจากชื่อสาร ไม่ใช่จากเหตุผลที่ใช้ยา

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
''')
    table(['รายการ','จำนวน'],[(k,v) for k,v in labels.items() if not isinstance(v,list)])
    add(f'''ผู้เข้าร่วมร่วมกันที่เปลี่ยนจาก positive เดิมเป็น negative ใหม่มี {labels['common_old_positive_new_negative']} คน
ไม่ควรเรียกคนเหล่านี้ว่า “ยืนยันว่าไม่เป็นเบาหวาน”
เพียงแต่ label เดิมกับ proxy ใหม่ใช้หลักฐานต่างกัน
มี {a['lbxs_gl_126_without_corrected_positive']} คนใน cohort หลังตัด known pregnancy ที่ LBXSGL>=126 แต่ไม่เข้า positive rule ใหม่
จำนวนนี้ไม่เท่ากับจำนวนคนที่เปลี่ยน label เสมอ เพราะบางคนถูกตัดเป็น indeterminate
หลักฐาน positive แต่ละชนิดด้านล่างนับทับซ้อนกันได้
''')
    table(['positive evidence','จำนวนผู้เข้าเงื่อนไขก่อนตัด label ไม่ชัด'],a['label_evidence_counts_overlapping'].items())
    add('''## Cleaning และ feature engineering

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
''')
    code('''df['bmi'] = df.weight_kg / (df.height_cm / 100).pow(2)
df['waist_to_height_ratio'] = df.waist_cm / df.height_cm
# ตัวอย่าง 90 kg / 1.72^2 = 30.4219... kg/m²
# train และ serving ใช้สูตรเดียวกัน; ปัดเศษเฉพาะตอนแสดงผล''')
    add('''ใช้ BMI ที่คำนวณจากน้ำหนักและส่วนสูงโดยตรงทั้ง train และ serving
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
''')
    features=[
        ('age','RIDAGEYR','ปี; อายุ 20+ และ 80 เป็น top-code','clip(upper=80)','อายุจริงตั้งแต่ 80 ขึ้นไปไม่สามารถแยกได้จากข้อมูลนี้'),
        ('bmi','BMXWT, BMXHT','kg/m²','weight_kg / (height_cm/100)^2','มีความสัมพันธ์ซ้ำกับน้ำหนักและส่วนสูง'),
        ('waist_cm','BMXWAIST','cm','ใช้ค่าบวกที่สังเกตจริง','หายแล้ว impute; ไม่มีสูตรเดารอบเอว'),
        ('height_cm','BMXHT','cm','ใช้ค่าบวกที่สังเกตจริง','เป็นตัวหารของ BMI และ waist/height'),
        ('weight_kg','BMXWT','kg','ใช้ค่าบวกที่สังเกตจริง','what-if คำนวณ BMI ใหม่ด้วยน้ำหนักนี้'),
        ('waist_to_height_ratio','BMXWAIST, BMXHT','ไม่มีหน่วย','waist_cm / height_cm','ไม่ใช้เป็นเหตุผลว่าลดเอวแล้วโรคลดลงเท่า SHAP'),
        ('systolic_bp','BPXSY1–3','mmHg','mean ของ readings >0','missing เมื่อไม่มี reading ใช้ได้'),
        ('diastolic_bp','BPXDI1–3','mmHg','mean ของ readings >0','zero readings ถูกเว้นออกตามนิยามโปรเจค'),
        ('daily_sugar_g','DR1TSUGR','g จาก recall วันแรก','ใช้เมื่อ DR1DRSTZ=1 และค่า>=0','เป็น total sugar ไม่ใช่ added sugar'),
        ('daily_fiber_g','DR1TFIBE','g จาก recall วันแรก','ใช้เมื่อ DR1DRSTZ=1 และค่า>=0','fiber=0 ยังเก็บ 0; ratio จะ missing'),
        ('daily_calories_kcal','DR1TKCAL','kcal จาก recall วันแรก','ใช้เมื่อ DR1DRSTZ=1 และค่า>=0','ถ้า UI ไม่ส่งค่า model จะ impute อย่างเปิดเผย'),
        ('sugar_to_fiber_ratio','sugar/fiber','อัตราส่วน','sugar / fiber เฉพาะ fiber>0','ไม่มีการบวก epsilon ที่ทำให้เกิดค่าหลายล้าน'),
        ('gender_is_female','RIAGENDR','binary','1→0, 2→1','สะท้อน coding ของ survey; API ต้องระบุ male/female'),
        ('hypertension_flag','BPQ020 และ BP summary','binary หรือ missing','history yes OR systolic>=140 OR diastolic>=90','คง history เมื่อจำลองว่าความดันปัจจุบันลดลง'),
        ('no_vigorous_recreation_flag','PAQ650','binary หรือ missing','yes→0, no→1','ห้ามเรียกเป็น sedentary flag'),
        ('smoking_history_flag','SMQ020','binary หรือ missing','yes→1, no→0','ประวัติ 100 มวน ไม่ใช่สถานะสูบปัจจุบัน'),
        ('sleep_trouble_flag','SLQ050','binary หรือ missing','yes→1, no→0','ประวัติเคยบอกแพทย์เรื่องการนอน'),
    ]
    for name,raw,units,formula,note in features:
        stats=m['feature_stats'][name]
        add(f'''### `{name}`

- ตัวแปรต้นทาง: `{raw}`
- หน่วย/ชนิด: {units}
- กฎคำนวณ: `{formula}`
- ข้อควรเข้าใจ: {note}
- Median ของ fit partition: {stats['median']:.6g}
- Mean ของ fit partition: {stats['mean']:.6g}
- ช่วงที่สังเกตใน fit: {stats['min']:.6g} ถึง {stats['max']:.6g}
- จำนวน missing ใน fit: {stats['missing_n']} แถว ({eda['missing_percentages'][name]:.2f}%)

Median ด้านบนใช้เพื่ออธิบายข้อมูลชุด fit
ตัวเติมค่าจริงภายในแต่ละ CV fold ยังต้อง fit ใหม่จาก fold-training ของตนเอง
''')
    add('''## การแบ่งข้อมูลและป้องกัน leakage

### บทบาทของแต่ละ partition

แบ่ง test=20% ออกก่อนด้วย stratification ตาม target
จาก development=80% แบ่ง calibration=25% ของ development
จึงได้สัดส่วนรวมโดยประมาณ fit 60% / calibration 20% / test 20%
ทั้งสองครั้งใช้ random_state=42
จำนวนจริงอาจไม่ตรงเปอร์เซ็นต์พอดีเพราะต้องเป็นจำนวนเต็ม
ใช้ SEQN เป็น index และตรวจ set ของแต่ละ partition ว่าไม่ทับกัน
''')
    table(['partition','n','positive','negative','positive fraction'],[(k,v['n'],v['positive'],v['n']-v['positive'],f"{v['positive']/v['n']:.4%}") for k,v in manifest['partitions'].items()])
    code('''X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
)''')
    add('''fit ใช้สร้างโมเดล ทำ CV เลือก champion และสร้าง OOF responses
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
''')
    code('''model = Pipeline([
    ('preprocessor', build_preprocessor()),
    ('classifier', clone(classifier)),
])
model.fit(X_fit.iloc[train_idx], y_fit.iloc[train_idx])
p = model.predict_proba(X_fit.iloc[val_idx])[:, 1]''')
    add('''### Imputer และ scaler ทำอะไร

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
''')
    table(['feature','missing % ใน fit','Pearson correlation กับ proxy ใน fit'],[(c,f"{eda['missing_percentages'][c]:.2f}",f"{eda['feature_correlations'][c]:+.4f}") for c in FEATURE_COLUMNS])
    add('''### ความหมายของ correlation

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
''')
    table(['โมเดล','CV ROC-AUC mean ± SD','CV AP mean ± SD','CV recall@0.5','CV F1@0.5'],[(n,f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}",f"{r['average_precision_mean']:.4f} ± {r['average_precision_std']:.4f}",f"{r['recall_at_0_5_mean']:.4f}",f"{r['f1_at_0_5_mean']:.4f}") for n,r in m['cv_benchmarks'].items()])
    add(f'''เลือก **{m['model_name']}** เพราะมี mean CV ROC-AUC สูงสุดในกลุ่มโมเดลที่ใช้ features
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
''')
    code('''calibrated = CalibratedClassifierCV(
    FrozenEstimator(raw), method='sigmoid', ensemble=False
)
calibrated.fit(X_cal, y_cal)
p_test = calibrated.predict_proba(X_test)[:, 1]''')
    add('''การมี calibration layer ไม่ได้เป็นหลักประกันว่าโมเดล calibrated สมบูรณ์
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
''')
    code('''fpr, tpr, thresholds = roc_curve(y_fit, mapped_oof, drop_intermediate=False)
valid = np.flatnonzero((tpr >= 0.80) & np.isfinite(thresholds))
i = min(valid, key=lambda k: (fpr[k], -thresholds[k]))
threshold = float(thresholds[i])
y_pred = (p_test >= threshold).astype(int)''')
    add(f'''threshold ของ champion = **{m['decision_threshold']:.12f}** บนสเกล 0–1
หรือประมาณ **{m['decision_threshold']*100:.3f}%** บนหน้าจอ
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
''')
    table(['โมเดล','threshold','Test AUC','Test AP','Sensitivity','Specificity','Precision','F1','Accuracy'],[(n,f"{r['threshold']:.6f}",*[f"{r[k]:.4f}" for k in ['roc_auc','average_precision','sensitivity','specificity','precision','f1','accuracy']]) for n,r in m['test_benchmarks'].items()])
    add('''### ตาราง test ที่ threshold=0.5 เพื่อดูผลของ operating point

โมเดลอาจมี AUC ใกล้เคียงกันแต่ recall ต่างกันมากเมื่อใช้ threshold เดียวกัน
การเลือก threshold เป็นส่วนสำคัญของ screening workflow
การเพิ่ม recall มักแลกกับจำนวน FP ที่สูงขึ้น
ไม่มีการบังคับให้ตัวเลข test recall ต้องเกิน 80% ทุกโมเดลหรือทุกกลุ่ม
''')
    table(['โมเดล','Recall@0.5','Specificity@0.5','Precision@0.5','F1@0.5','Accuracy@0.5'],[(n,*[f"{r['at_0_5'][k]:.4f}" for k in ['sensitivity','specificity','precision','f1','accuracy']]) for n,r in m['test_benchmarks'].items()])
    add('''### Calibration บน final test: raw เทียบกับ calibrated

Brier และ log loss คำนวณจาก probabilities ไม่ขึ้นกับ classification threshold
การเปลี่ยน threshold ไม่สามารถทำให้ Brier ดีขึ้นได้
ผล calibration ไม่จำเป็นต้องทำให้ ROC-AUC ดีขึ้น เพราะ sigmoid ที่ monotonic มักรักษา ranking
''')
    table(['โมเดล','Raw Brier','Calibrated Brier','Raw log loss','Calibrated log loss'],[(n,f"{r['raw_uncalibrated']['brier']:.5f}",f"{r['brier']:.5f}",f"{r['raw_uncalibrated']['log_loss']:.5f}",f"{r['log_loss']:.5f}") for n,r in m['test_benchmarks'].items()])
    model_explanations={
        'DummyPrior':('คืนค่า class prior โดยไม่ใช้ features','เป็น baseline ว่าใช้ ML แล้วดีกว่าการทาย prevalence คงที่หรือไม่','AUC=0.5 และ AP ใกล้ positive fraction; threshold ที่บังคับ recall>=80% ทำให้ทาย positive ทั้งหมดได้'),
        'LogisticRegression':('เรียน linear relationship บน log-odds พร้อม L2 regularization','C=0.5, balanced class weights, max_iter=3000','ตีความ coefficient ต้องคำนึงถึง RobustScaler และ features ที่สัมพันธ์กัน; ใช้ sigmoid calibration ครอบอีกชั้น'),
        'RandomForest':('bagging ของ decision trees พร้อมสุ่ม features','250 trees, max_depth=8, min_samples_split=10, min_samples_leaf=5, balanced','เป็น champion ตาม CV; มี nonlinear interactions แต่ correlated features ทำให้ importance กระจาย'),
        'ExtraTrees':('ensemble ของต้นไม้ที่สุ่ม split thresholds มากขึ้น','250 trees, max_depth=8, min_samples_split=10, min_samples_leaf=5, balanced','ความสุ่มอาจลด variance แต่ไม่รับประกันว่าจะชนะ Random Forest'),
        'GradientBoosting':('เพิ่มต้นไม้แบบ sequential เพื่อแก้ residual/errors','180 stages, learning_rate=0.04, max_depth=4, subsample=0.85','ไม่มี class_weight ใน candidate นี้; recall@0.5 จึงเปรียบเทียบกับ balanced raw models อย่างผิวเผินไม่ได้'),
        'HistGradientBoosting':('boosting โดยจัด numerical features เป็น bins','max_iter=180, learning_rate=0.04, max_depth=5, balanced','เป็น fixed configuration; ไม่ใช่ผลที่ผ่าน tuning exhaustive search'),
    }
    for name,r in m['test_benchmarks'].items():
        desc,params,note=model_explanations[name]
        cv=m['cv_benchmarks'][name]
        cm=r['confusion_matrix']
        add(f'''### วิเคราะห์ `{name}` รายตัว

กลไก: {desc}

พารามิเตอร์หลัก: {params}

ข้อควรเข้าใจ: {note}

ผล CV ก่อน calibration:

- ROC-AUC mean = {cv['roc_auc_mean']:.6f}
- ROC-AUC SD = {cv['roc_auc_std']:.6f}
- Average Precision mean = {cv['average_precision_mean']:.6f}
- Recall@0.5 mean = {cv['recall_at_0_5_mean']:.6f}

ผล test หลัง calibration และใช้ threshold={r['threshold']:.6f}:

- ROC-AUC = {r['roc_auc']:.6f}
- AP = {r['average_precision']:.6f}
- Sensitivity = {r['sensitivity']:.6f}
- Specificity = {r['specificity']:.6f}
- Precision = {r['precision']:.6f}
- F1 = {r['f1']:.6f}
- Accuracy = {r['accuracy']:.6f}
- Brier = {r['brier']:.6f}
- Log loss = {r['log_loss']:.6f}

Confusion matrix ใช้ลำดับแถว true 0/1 และคอลัมน์ predicted 0/1:
''')
        table(['','predicted 0','predicted 1'],[['true 0',cm['tn'],cm['fp']],['true 1',cm['fn'],cm['tp']]])
        add('ผลแต่ละ fold อยู่ด้านล่างเพื่อให้เห็นความผันผวนและตรวจค่าเฉลี่ยเองได้')
        table(['fold','n fit','n validation','AUC','AP','Recall@0.5','F1@0.5'],[(f['fold'],f['n_fit'],f['n_validation'],*[f"{f[k]:.6f}" for k in ['roc_auc','average_precision','recall_at_0_5','f1_at_0_5']]) for f in cv['folds']])
    champion=m['test_benchmarks'][m['model_name']]
    cm=champion['confusion_matrix']
    add(f'''### ตีความผล champion เป็นจำนวนคน

Test มี {m['test_n']} คน
ในนี้ proxy positive {cm['tp']+cm['fn']} คน และ proxy negative {cm['tn']+cm['fp']} คน
ตรวจพบ positive ได้ {cm['tp']} คน แต่พลาด {cm['fn']} คน
ใน proxy negative ทายลบถูก {cm['tn']} คน และทายบวกผิด {cm['fp']} คน
Precision={champion['precision']:.2%} หมายถึง positive predictions ส่วนใหญ่ยังไม่ใช่ proxy positive
นี่เป็นข้อแลกเปลี่ยนของการเลือก threshold ที่เน้น sensitivity

Accuracy ของ DummyPrior ที่ threshold 0.5 คือ {m['test_benchmarks']['DummyPrior']['at_0_5']['accuracy']:.2%}
อาจสูงกว่า accuracy ของ screening operating point ทั้งที่ไม่มีความสามารถจัดอันดับเลย
จึงไม่ควรเลือกโมเดลด้วย accuracy เพียงค่าเดียวเมื่อ class imbalance และต้นทุน FN/FP ต่างกัน

## ความไม่แน่นอนและผลแยกกลุ่ม

### ช่วงความไม่แน่นอนของ final test

ใช้ stratified bootstrap 1,000 รอบ seed=42
สุ่มด้วย replacement แยกใน true class 0 และ true class 1
คงจำนวนแต่ละ class เท่ากับ observed test ทุกครั้ง
คำนวณ AUC/AP/sensitivity/specificity ในแต่ละรอบ
ใช้ quantiles 2.5% และ 97.5% เป็น percentile interval
''')
    table(['โมเดล','AUC 95% interval','AP 95% interval','Sensitivity 95% interval','Specificity 95% interval'],[(n,*[f"[{r['ci95_stratified_iid_bootstrap'][k][0]:.4f}, {r['ci95_stratified_iid_bootstrap'][k][1]:.4f}]" for k in ['roc_auc','average_precision','sensitivity','specificity']]) for n,r in m['test_benchmarks'].items()])
    add('''intervals นี้ conditional on fitted model และ observed class counts
ไม่ได้รวมความผันผวนจากการ train ใหม่ การเลือกโมเดล หรือการประมาณ prevalence ใหม่
ไม่ได้ปรับ survey weights, strata หรือ PSU clustering
จึงไม่ใช่ design-based NHANES confidence interval
ไม่ได้ใช้เพื่อประกาศความเหนือกว่าระหว่างโมเดลอย่างเป็นทางการ
การเห็น intervals ทับกันไม่ใช่ formal hypothesis test และการไม่ทับกันก็ต้องดูวิธีเปรียบเทียบให้ถูก

### ตรวจผลแยกกลุ่มของ champion

ตารางนี้เป็น descriptive diagnostic หลังล็อกโมเดลแล้ว
ใช้ threshold เดิมทุกกลุ่ม ไม่ได้ปรับ threshold ให้กลุ่มใดหลังดูผล
ไม่ใช่ fairness certification และจำนวน positive บางกลุ่มน้อย
''')
    table(['กลุ่ม','n','positive','AUC','AP','Sensitivity','Specificity'],[(n,r['n'],r['positive'],*[f"{r['metrics'][k]:.4f}" for k in ['roc_auc','average_precision','sensitivity','specificity']]) for n,r in subgroups.items()])
    add('''จุดที่ควรอ่านเป็นพิเศษคือกลุ่มอายุ 20–39 มี positive เพียง 11 คน
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
''')
    code('''p = calibrated_model.predict_proba(X)[:, 1]
explanation = explainer(X.to_numpy(), max_evals=35, silent=True)
reconstructed = explanation.base_values + explanation.values.sum(axis=1)
np.testing.assert_allclose(reconstructed, p, atol=1e-7)
# example: base 0.15 + contributions 0.10 = probability 0.25
# display: base 15 pp + total contribution 10 pp = score 25%''')
    add('''API ตรวจ sum(base + contributions) ให้ตรงกับ probability ก่อนตอบ
คืน shap_base_score และ shap_total_contribution_points ไว้ตรวจย้อนกลับ
หน้าจอแสดงเฉพาะ top 3 ที่เพิ่ม score และ top 2 ที่ลด score
จึงไม่ควรบวกเฉพาะ 5 รายการบนหน้าจอแล้วคาดว่าจะได้ score เต็ม
ไม่มี factor ติดลบก็คืนรายการว่าง ไม่สร้างคำแนะนำปลอมเติมเข้าไป
field direction ใช้ legacy convention: negative หมายถึงเพิ่ม score และ positive หมายถึงลด score
ควรดูเครื่องหมายของ impact_value ประกอบเพราะชื่อ direction เดิมอาจชวนสับสน

### Global importance ที่คำนวณจริง
''')
    table(['อันดับ','feature','mean(abs(SHAP)) probability units','เทียบเป็น pp'],[(i,r['feature'],f"{r['importance']:.6f}",f"{r['importance']*100:.4f}") for i,r in enumerate(m['feature_importance'],1)])
    add('''### ข้อจำกัดของ feature attribution

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
''')
    table(['method','path','หน้าที่'],[
        ['GET','/health หรือ /api/health','ตรวจ server และการโหลดโมเดล'],
        ['GET','/api/metadata','metrics, threshold, feature stats, importance, limitations'],
        ['POST','/api/predict','calibrated score และ local associations'],
        ['POST','/api/simulate','เปลี่ยน input แล้วคำนวณ score ของ scenario'],
    ])
    add('''### ตัวอย่าง request/response ที่เรียก API จริง

ตัวอย่างด้านล่างเก็บจาก TestClient ผ่าน route ของ FastAPI จริง
score เป็นผลจาก artifact ที่สร้างใน audit นี้
SHAP ราย feature อาจเปลี่ยนเล็กน้อยเมื่อเรียกใหม่เพราะใช้ permutation approximation
probability ของ input เดิมยังคงเดิม
''')
    for name in ['predict','simulate','identity_simulation']:
        ex=examples[name]
        add(f"#### `{name}` — `{ex['path']}`\n\nRequest:")
        code(json.dumps(ex['request'],indent=2,ensure_ascii=False),'json')
        add('Response ที่ได้จริง:')
        code(json.dumps(ex['response'],indent=2,ensure_ascii=False),'json')
    add('''## What-if และ frontend

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
''')
    table(['package','version ที่ใช้'],manifest['versions'].items())
    add('''frontend build ตรวจพบ Next.js 14.2.35 ตาม package-lock ที่มีอยู่
README ระบุ version ที่ใช้จริง ไม่ได้หมายความว่าทุก version อื่นจะรันไม่ได้
สำหรับ reproducibility ของ joblib ควรใช้ Python และ library versions ตรงกัน
ไฟล์ requirements-lock.txt เก็บ exact versions ของ Python dependencies ที่ใช้
ตัด LightGBM/XGBoost ที่ติดตั้งเดิมแต่ไม่ได้ใช้ออกจาก lock เพื่อไม่บังคับติดตั้ง runtime ที่ไม่เกี่ยวข้อง

### 1. สร้าง environment และติดตั้ง
''')
    code('''cd "/path/to/CPE310 AI"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
# หากต้องการช่วงเวอร์ชันกว้างแทน exact reproduction:
# python -m pip install -r requirements-dev.txt''','bash')
    add('''ต้องมี scikit-learn>=1.6 เพราะใช้ FrozenEstimator
ไม่ต้องใช้ GPU, AWS, external database หรือ API key
ไม่ต้องดาวน์โหลด dataset ใหม่เพราะมี CSV ทั้ง 6 ไฟล์อยู่แล้ว
เมื่อ install ต่างเวอร์ชันควร retrain/re-export แทนการเชื่อว่า joblib เก่า portable เสมอ

### 2. เตรียมข้อมูลและดู cohort
''')
    code('python -m ml_pipeline.preprocess','bash')
    add('''คำสั่งนี้อ่าน CSV, merge, label และ engineer features
พิมพ์ data audit กับจำนวน missing
ยังไม่ train model และยังไม่แก้ไข raw CSV

### 3. แบ่งข้อมูลและ train/CV
''')
    code('python -u -m ml_pipeline.train','bash')
    add('''สคริปต์สร้าง fit/calibration/test partitions
รัน 5-fold CV ของ candidate ทั้ง 6
fit final base model ของแต่ละ candidate บน fit partition
บันทึก split, OOF predictions, base models และ manifest
SEQN ของทุก partition อยู่ใน run_manifest.json เพื่อตรวจ patient overlap

### 4. สร้าง EDA ของ fit
''')
    code('python -m ml_pipeline.eda','bash')
    add('''ต้อง train/split ก่อนจึงค่อยเรียก eda.py
ต่างจากคำสั่งใน README เดิมที่เรียก EDA ก่อนมี split
ตัวเลข EDA ใหม่จะเป็น fit-only ตาม protocol ที่แก้ไข

### 5. Calibrate, evaluate และ export
''')
    code('python -u -m ml_pipeline.evaluate','bash')
    add('''อ่าน fitted models ที่ล็อกไว้
เรียน calibrator บน calibration partition
เลือก thresholds จาก fit OOF mapped scores
ประเมิน test ของทุก candidate พร้อม bootstrap intervals
สร้าง SHAP artifact สำหรับ champion และตรวจ additivity
export serving model และ metadata ชุดเดียวกัน
หากข้ามขั้นตอนนี้ backend จะไม่มี artifact schema 2 ที่สอดคล้องกับโค้ดใหม่
ต้อง restart backend หลัง re-export เพื่อไม่ให้ process ใช้โมเดลที่โหลดค้างอยู่

### 6. สร้าง audit comparisons และ API examples
''')
    code('python -m ml_pipeline.audit_report','bash')
    add('''ขั้นตอนนี้ใช้ original source ที่สำรองไว้เพื่อเทียบ label
ไม่ได้ train หรือเลือกโมเดลใหม่
เรียก routes ผ่าน TestClient และบันทึก JSON request/response จริง
subgroup metrics เป็นการตรวจเชิงอธิบายหลังล็อกโมเดล

### 7. ตรวจ regression tests
''')
    code('python -m unittest discover -s tests -v','bash')
    add('''### 8. สร้าง README จากผลล่าสุด
''')
    code('python -m ml_pipeline.document','bash')
    add('''คำสั่งนี้เขียน README.md ทับจาก artifacts และ source ปัจจุบัน
อย่ารันก่อนมี metadata/manifest/EDA/audit examples ครบ
หากเปลี่ยน source ให้รันทดสอบส่วนที่เกี่ยวข้องแล้วสร้าง README ใหม่
ไม่ต้องแก้ตารางผลโมเดลด้วยมือทีละค่า

### 9. เปิด backend
''')
    code('python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload','bash')
    add('''เปิด Swagger UI ที่ http://localhost:8000/docs
ตรวจ health ที่ http://localhost:8000/health
ใช้ 127.0.0.1 สำหรับการทดลองในเครื่อง
configuration ปัจจุบันเป็น local-development configuration ไม่ได้ผ่าน production deployment review

### 10. เปิด frontend ใน terminal อีกแท็บ
''')
    code('''cd "/path/to/CPE310 AI/frontend"
npm ci
npm run dev''','bash')
    add('''เปิด http://localhost:3000
กรอกข้อมูลจริงที่ทราบ ปล่อย optional measurements ว่างเมื่อไม่ทราบ
ตรวจรายการ imputed fields เมื่ออ่านผล
อย่าใช้ผลจาก preset เป็นหลักฐานว่าผู้ใช้จริงต้องอยู่ใน risk tier เดียวกัน

### 11. ตรวจ TypeScript และ production build
''')
    code('''cd "/path/to/CPE310 AI/frontend"
./node_modules/.bin/tsc --noEmit
npm run build''','bash')
    add('''## การทดสอบและตรวจซ้ำ

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
''')
    table(['test','ความผิดพลาดที่ป้องกัน'],[
        ['target rule','ไม่ใช้ serum glucose>=126 เป็น positive; missing A1c ไม่เป็น control; therapy/diagnosis override'],
        ['medication parser','ไม่จับ glucosamine หรือยาที่ indication กล่าวถึง diabetes อย่างเดียว'],
        ['hypertension unknown','code 7/9 ไม่กลายเป็น no history'],
        ['CV fit boundary','ตรวจจำนวนแถว fit ต่อ fold และ median ที่เรียนจาก fold-training'],
        ['threshold/confusion matrix','ตรวจ max-specificity subject to recall rule บน toy case ที่รู้คำตอบ'],
        ['subject partitions','SEQN ไม่ทับกันและ serving statistics เป็น fit-only'],
        ['train/serve parity','นำแถวจริงมาแปลงผ่าน API แล้วได้ features เดียวกับ training'],
        ['SHAP output','sum ตรวจย้อน probability; signs/units ตรงและ imputed calorie เปิดเผย'],
        ['missing inputs','ไม่มีรอบเอวหรือ BP ก็ไม่สร้าง measured value ปลอม'],
        ['zero fiber/validation','ratio missing ได้, age<20 ถูกปฏิเสธ, gender ต้องส่ง'],
        ['simulation identity/history','ไม่มี adjustments ต้อง delta=0; BP scenario ไม่ลบ diagnosis history'],
        ['exported metrics','คำนวณ test metrics ทุกโมเดลใหม่จาก CSV และตรวจ champion rule'],
    ])
    add('''### วิธีคำนวณ test metrics ซ้ำด้วยตนเอง
''')
    code('''import json
import pandas as pd
from ml_pipeline.evaluate import classification_metrics

meta = json.load(open('ml_pipeline/artifacts/metadata.json'))
rows = pd.read_csv('ml_pipeline/artifacts/test_predictions.csv')
for name, frame in rows.groupby('model'):
    threshold = meta['test_benchmarks'][name]['threshold']
    result = classification_metrics(frame.y_true, frame.probability, threshold)
    print(name, result['roc_auc'], result['confusion_matrix'])''')
    add('''### วิธีตรวจว่า test ไม่ได้กำหนด champion
''')
    code('''import json
meta = json.load(open('ml_pipeline/artifacts/metadata.json'))
best_cv = max(
    (n for n in meta['cv_benchmarks'] if n != 'DummyPrior'),
    key=lambda n: meta['cv_benchmarks'][n]['roc_auc_mean'],
)
assert best_cv == meta['model_name']
# อย่าเปลี่ยนเป็น max(test_benchmarks, key=...test ROC-AUC...)''')
    add('''### artifacts แต่ละไฟล์ใช้ตรวจอะไร
''')
    table(['artifact','รายละเอียด'],[
        ['run_manifest.json','seed, split SEQNs, source hashes, counts, versions, selection rule'],
        ['train_test_splits.joblib','X/y สาม partitions, fit-only stats, OOF, fitted base models'],
        ['benchmark_models.joblib','calibrated candidate models ทุกตัวที่ใช้รายงาน test'],
        ['model_pipeline.joblib','calibrated champion ที่ API เรียกจริง'],
        ['shap_explainer.joblib','explainer, feature names, base value, probability output space'],
        ['metadata.json','CV/test metrics, thresholds, calibration notes, limitations, serving statistics'],
        ['test_predictions.csv','หนึ่งแถวต่อคนต่อโมเดล; y_true, p, raw_p, threshold, predicted class'],
        ['champion_oof_predictions.csv','OOF probability ที่ใช้ threshold/reference ของ champion'],
        ['evaluation_curves.json','ROC, PR และ quantile calibration curve arrays ทุกโมเดล'],
        ['eda_summary.json','fit-only correlations, group means, missingness, statistics'],
    ])
    add('''joblib เป็น Python serialization format ที่มีโค้ด/objects อยู่ได้
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
''')
    modules=[
        ('ml_pipeline/preprocess.py','การอ่านข้อมูล นิยาม cohort/label และ features',[
            '`read_table`: ตรวจ schema และ primary key ก่อน merge',
            '`load_and_merge_data`: filter adults/known pregnancy แล้ว left join แบบ one-to-one',
            '`get_diabetic_medication_seqns`: aggregate ingredient evidence จากหลายแถวต่อคน',
            '`define_target`: positive/negative/indeterminate โดยไม่ impute outcome',
            '`safe_sugar_fiber_ratio`: ทำให้ zero-denominator เป็น missing แทนค่ามหาศาล',
            '`hypertension_indicator`: ไม่ให้ unknown questionnaire กลายเป็น false',
            '`engineer_features`: สูตรที่ train และ inference ใช้ร่วมกัน',
            '`describe_features`: summary เท่านั้น; train.py เลือกส่งเฉพาะ fit partition เข้าไปสำหรับ serving',
            '`build_preprocessor`: median/mode/RobustScaler ที่จะ fit ภายในแต่ละ fold',
            '`get_prepared_data`: คืน X/y ที่ index เป็น SEQN และข้อมูล audit']),
        ('ml_pipeline/train.py','การแบ่งข้อมูล ทำ CV และเลือก base model',[
            '`get_candidate_models`: candidate configurations ที่กำหนดตายตัว',
            '`build_model`: clone classifier และสร้าง preprocessor ใหม่ทุกครั้ง',
            '`response_scores`: จัดรูป response ให้ตรงวิธี calibrate ของ sklearn',
            '`benchmark_models`: fit/validate ทีละ fold, เก็บ OOF และ fold metrics',
            '`train_and_export`: split 60/20/20, ตรวจ overlap, fit-only stats และ manifest']),
        ('ml_pipeline/evaluate.py','Calibration, threshold, test และ export',[
            '`classification_metrics`: confusion matrix และ metrics ทั้ง threshold-dependent/independent',
            '`select_threshold`: บังคับ sensitivity target ก่อน maximize specificity',
            '`bootstrap_intervals`: uncertainty แบบ stratified IID conditional bootstrap',
            '`run_evaluation`: ใช้ champion ที่เลือกไว้แล้ว, freeze base estimator, calibrate และทดสอบ',
            'ส่วน SHAP ครอบ deployed model จริง ไม่เลือกอธิบายเฉพาะ estimator ภายใน',
            'test_predictions.csv และ metadata ใช้ค่าชุดเดียวกัน ไม่ hardcode จำนวน test 1,131']),
        ('ml_pipeline/explain.py','Callable สำหรับอธิบาย calibrated probabilities',[
            '`ProbabilityPredictor`: แปลง NumPy ที่ SHAP ส่งมาเป็น DataFrame ตาม feature order',
            '`compute_sample_shap`: คืน contributions ตามชื่อ features ของ request หนึ่งแถว',
            'การ define callable ใน module ทำให้ joblib deserialize ได้เมื่อ backend โหลด']),
        ('ml_pipeline/eda.py','EDA หลังแบ่งข้อมูลแล้ว',[
            'อ่าน split artifact แทนการ summarize target ของ test',
            'คำนวณ correlation, group means และ missingness เฉพาะ fit partition',
            'ไม่ได้ใช้ผลนี้เลือก feature เพิ่มหลังดู test']),
        ('backend/app/schemas.py','ข้อตกลง input/output ของ API',[
            '`AssessmentInput`: bounds, optional missing fields, age>=20 และ gender ที่ต้องส่ง',
            '`FeatureImpact`: SHAP contribution ใน probability units และข้อความ pp',
            '`PredictionResponse`: score, threshold result, missing fields และ additivity summary',
            '`SimulationDelta`: ช่วง delta ที่อนุญาตและ nullable activity override',
            '`SimulationResponse`: คะแนนเดิม/ใหม่และการเปลี่ยน score']),
        ('backend/app/model_service.py','Serving ที่ใช้สูตรเดียวกับ training',[
            '`_input_to_dataframe`: map request กลับ raw-like columns แล้วใช้ engineer_features',
            '`predict`: calibrated output, categories, OOF percentile, SHAP และ additivity check',
            '`simulate`: apply deltas, preserve history, clamp bounds และคำนวณ scenario',
            'ทุก score-decreasing factor มาจาก contribution ติดลบจริง']),
        ('tests/test_pipeline.py','Tests ที่รันจริงสำหรับ invariants และ integration',[
            'มีทั้งตัวอย่างที่ทราบคำตอบและการตรวจ artifact จริง',
            'ตรวจ CV fit boundary ผ่าน recording pipeline โดยไม่ต้อง retrain candidate ใหญ่',
            'ตรวจ API ผ่าน FastAPI routes ไม่ได้เรียกเพียง helper function']),
        ('frontend/src/components/WhatIfSimulator.tsx','State และ input scenarios ของหน้าเว็บ',[
            'เริ่มต้นด้วย adjustments ศูนย์และ activity เท่ากับ baseline',
            'request ID ป้องกัน response เก่าทับ response ใหม่เมื่อเลื่อนเร็ว',
            'แสดง score difference เป็น pp พร้อมข้อความ association-only',
            'reset คืน state ให้ตรงกับ baseline รวม activity']),
    ]
    for path,title,notes in modules:
        add(f'### `{path}` — {title}')
        add('\n'.join('- '+note for note in notes))
        code((ROOT/path).read_text(), 'tsx' if path.endswith('.tsx') else 'python')
    add('''## แหล่งอ้างอิงหลัก

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
''')
    result='\n'.join(chunks)
    (ROOT/'README.md').write_text(result,encoding='utf-8')
    print(f'README: {len(result.splitlines())} lines, {len(result.encode())} bytes')


if __name__=='__main__':
    generate()
