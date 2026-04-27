import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. استيراد البيانات
df = pd.read_csv('medical_examination.csv')

# 2. إضافة عمود 'overweight' (سمنة مفرطة)
# حساب مؤشر كتلة الجسم BMI: الوزن (كجم) / مربع الطول (متر)
# إذا كان BMI > 25 فالقيمة 1، وإلا 0
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3. تطبيع البيانات (Normalize data)
# جعل 0 دائماً جيد و 1 دائماً سيء للـ cholesterol و gluc
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. رسم المخطط الفئوي (Categorical Plot)
def draw_cat_plot():
    # 5. إنشاء DataFrame للمخطط باستخدام pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. تجميع البيانات وإعادة تنسيقها لإظهار التكرارات
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. رسم المخطط باستخدام seaborn.catplot()
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 8. حفظ الصورة
    fig.savefig('catplot.png')
    return fig

# 9. رسم الخريطة الحرارية (Heat Map)
def draw_heat_map():
    # 10. تنظيف البيانات في df_heat حسب المعايير المطلوبة
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 11. حساب مصفوفة الارتباط
    corr = df_heat.corr()

    # 12. إنشاء "قناع" (mask) للمثلث العلوي للمصفوفة
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13. إعداد شكل الرسم (matplotlib figure)
    fig, ax = plt.subplots(figsize=(12, 12))

    # 14. رسم الخريطة الحرارية باستخدام sns.heatmap()
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5})

    # 15. حفظ الصورة
    fig.savefig('heatmap.png')
    return fig
