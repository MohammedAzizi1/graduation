from flask import Flask, render_template, request
from catboost import CatBoostClassifier
# import numpy as np # NumPy قد لا تكون ضرورية إذا كان CatBoost يتعامل مع القوائم مباشرة

app = Flask(__name__)

# تحميل نموذج CatBoost المدرب مسبقًا
MODEL_PATH = r"C:\Users\Mohammed_Azizi\Desktop\gradution\gradution - Copy\model.cbm" # استخدم r"" للمسارات لتجنب مشاكل الشرطة المائلة العكسية
try:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print(f"تم تحميل نموذج CatBoost بنجاح من: {MODEL_PATH}")
except Exception as e:
    print(f"خطأ أثناء تحميل نموذج CatBoost من '{MODEL_PATH}': {e}")
    model = None

@app.route('/')
def index():
    """يعرض صفحة الإدخال الرئيسية."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """يستقبل البيانات من النموذج، يقوم بالتنبؤ، ويعرض صفحة النتائج."""
    if model is None:
        return render_template('result.html',
                               prediction="Error",
                               probability=0,
                               error="Model not loaded, please check server logs.")

    try:
        # 1. الحصول على البيانات من النموذج مع قيم افتراضية وتحويلها لحالة صغيرة
        symptom_fields = [
            'polyuria', 'polydipsia', 'weight_loss', 'weakness',
            'polyphagia', 'genital_thrush', 'visual_blurring', 'itching',
            'irritability', 'delayed_healing', 'partial_paresis',
            'muscle_stiffness', 'alopecia', 'obesity'
        ]

        form_data = {}
        # تأكد أن 'name' في HTML لـ gender هو "gender"
        form_data['gender'] = request.form.get('gender', 'male').lower()
        # تأكد أن 'name' في HTML لـ age هو "age"
        form_data['age'] = request.form.get('age', '30') # قيمة افتراضية للعمر كسلسلة نصية

        for field in symptom_fields:
            # تأكد أن 'name' في HTML لكل عرض يطابق هذه الأسماء
            form_data[field] = request.form.get(field, 'no').lower()

        # طباعة البيانات المستلمة (للتصحيح)
        print("Form data received (with defaults and lowercase):", form_data)

        # 2. التحقق من صحة العمر وتحويله إلى رقم
        try:
            age_value = int(form_data['age'])
            if not (0 <= age_value <= 120):
                raise ValueError("Age must be between 0 and 120")
        except ValueError as ve:
            print(f"Age validation error: {ve}")
            return render_template('result.html',
                                   prediction="Invalid Input",
                                   probability=0,
                                   error=f"Please enter a valid age (0-120). Error: {ve}")

        # 3. تحويل البيانات إلى التنسيق الرقمي الذي يتوقعه النموذج
        # **مهم جدًا: هذا الترتيب يجب أن يطابق تمامًا ترتيب الميزات عند تدريب النموذج**

        # تحويل الجنس: افترض أن النموذج مدرب على Male=0, Female=1
        # إذا كان العكس، قم بتغيير (0 if ... else 1)
        gender_numeric = 0 if form_data['gender'] == 'male' else 1

        # قائمة بأسماء الأعراض بنفس الترتيب الذي تم تدريب النموذج عليه
        # عدّل هذه القائمة إذا كان ترتيب ميزات نموذجك مختلفًا
        expected_feature_order_from_training = [
            'Age', 'Gender', 'Polyuria', 'Polydipsia', 'Sudden_Weight_Loss',
            'Weakness', 'Polyphagia', 'Genital_Thrush', 'Visual_Blurring',
            'Itching', 'Irritability', 'Delayed_Healing', 'Partial_Paresis',
            'Muscle_Stiffness', 'Alopecia', 'Obesity'
        ]
        # ملاحظة: الأسماء أعلاه هي مثال. يجب أن تستخدم الأسماء الفعلية للميزات
        # التي تم استخدامها لتدريب نموذج CatBoost الخاص بك.
        # إذا كان نموذجك لا يتوقع أسماء الميزات ولكن فقط مصفوفة من القيم،
        # فإن الترتيب لا يزال هو الأهم.

        # إنشاء قائمة الميزات الرقمية بالترتيب الصحيح
        input_features_for_model = []

        # مثال لإنشاء قائمة المدخلات بناءً على الترتيب المفترض
        # **عدّل هذا الجزء ليتناسب مع ترتيب الميزات الدقيق لنموذجك**
        # لنفترض أن ترتيب نموذجك هو كما في القائمة 'expected_feature_order_from_training'
        # وأن الأعراض في ملف HTML لها أسماء مثل 'polyuria', 'weight_loss' إلخ.
        # وأن اسم عمود 'weight_loss' في التدريب كان 'Sudden_Weight_Loss'

        # هذا مثال، يجب تكييفه:
        # إذا كان ترتيب الميزات في التدريب: Age, Gender, Polyuria, Polydipsia, ...
        input_features_for_model.append(age_value)
        input_features_for_model.append(gender_numeric)
        input_features_for_model.append(1 if form_data['polyuria'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['polydipsia'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['weight_loss'] == 'yes' else 0) # اسم الحقل في HTML
        input_features_for_model.append(1 if form_data['weakness'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['polyphagia'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['genital_thrush'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['visual_blurring'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['itching'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['irritability'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['delayed_healing'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['partial_paresis'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['muscle_stiffness'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['alopecia'] == 'yes' else 0)
        input_features_for_model.append(1 if form_data['obesity'] == 'yes' else 0)


        # طباعة الميزات النهائية قبل التنبؤ (للتصحيح)
        print("Input features for model (ensure order matches training):", input_features_for_model)
        print(f"Number of features for model: {len(input_features_for_model)}")


        # 4. إجراء التنبؤ
        # CatBoost يتوقع قائمة من القوائم للمدخلات (حتى لو كان صفًا واحدًا)
        prediction_proba_raw = model.predict_proba([input_features_for_model])
        # الفئة 1 هي عادةً الإيجابية (خطر الإصابة)
        probability_positive = round(prediction_proba_raw[0][1] * 100, 2)

        # تحديد التصنيف بناءً على الاحتمالية أو model.predict()
        # model.predict() سيعطي 0 أو 1 مباشرة (افترض أن 1 = Positive)
        predicted_class = model.predict([input_features_for_model])[0]
        prediction_text = "Positive" if predicted_class == 1 else "Negative"

        print(f"Raw probabilities from model: {prediction_proba_raw}")
        print(f"Predicted class (0 or 1): {predicted_class}, Calculated positive probability: {probability_positive}%")


        return render_template('result.html',
                               prediction=prediction_text,
                               probability=probability_positive)

    except KeyError as ke:
        # هذا الخطأ يحدث إذا كان اسم الحقل المستخدم في request.form.get أو form_data[] غير موجود
        # أو إذا كان اسم الميزة في expected_feature_order_from_training غير متطابق مع form_data
        print(f"KeyError during prediction: {ke}. This means a form field name or feature name is mismatched.")
        return render_template('result.html',
                               prediction="Error",
                               probability=0,
                               error=f"Data processing error: Missing or mismatched field '{str(ke)}'. Please check form field names and feature order.")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        import traceback
        traceback.print_exc() # لطباعة تتبع الخطأ الكامل في الطرفية
        return render_template('result.html',
                               prediction="Error",
                               probability=0,
                               error=f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)