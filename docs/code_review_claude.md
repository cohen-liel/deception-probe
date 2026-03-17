
                                                                                                                        
  ---                                                                                                                   
  הרושם הכללי                                                                                                           
                                                                                                                        
  המחקר מרשים מאוד מבחינת עיצוב ניסויי. ה-same-prompt design (ניסוי 02) הוא התרומה המרכזית והחשובה ביותר — רוב העבודות  
  בתחום סובלות בדיוק מה-confound שאתם מתארים. הפרדה בין שקר להזיה (ניסוי 03) היא תוצאה מאוד מעניינת. מבנה הריפו נקי,
  מתועד היטב, וקל למעקב.

  אבל יש כמה באגים מתודולוגיים משמעותיים וכמה בעיות קוד שחייבים לתקן לפני פרסום:

  ---
  באגים קריטיים

  1. Data Leakage ב-train_probe (הבאג הכי חמור)

  src/utils.py:242-243:
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)  # ← BUG: fit on ALL data before CV

  ה-StandardScaler מאומן על כל הדאטה (כולל ה-test folds) לפני ה-cross-validation. זה data leakage קלאסי. ה-scaler רואה
  את ה-mean/std של הדאטה שאמור להיות "מוסתר" בכל fold. זה יכול לנפח את הדיוק.

  תיקון: צריך להכניס את ה-scaling לתוך pipeline או לעשות fit רק על ה-train fold:
  from sklearn.pipeline import Pipeline
  pipe = Pipeline([
      ('scaler', StandardScaler()),
      ('clf', LogisticRegression(max_iter=1000, ...))
  ])
  scores = cross_val_score(pipe, X, y, cv=cv, scoring=bal_scorer)

  2. חוסר עקביות ב-permutation_test vs train_probe

  src/utils.py:281:
  clf = LogisticRegression(max_iter=1000, random_state=random_seed, C=1.0)
  # ← חסר class_weight="balanced"!

  ב-train_probe יש class_weight="balanced", אבל ב-permutation_test אין. ה-null distribution נבנית עם classifier אחר מזה
  שנמדד. זה פוגע בתקפות של ה-p-value.

  3. check_answer_match — התאמה חלשה מדי

  src/utils.py:213-220:
  words = [w for w in answer_lower.split() if len(w) > 3]
  if words:
      return any(w in resp_lower for w in words)

  בודק אם מילה אחת בלבד (מעל 3 תווים) מופיעה בתשובה. דוגמה: אם התשובה הנכונה היא "Peter Principle" ותשובת המודל כוללת
  "Peter" בהקשר אחר — זה ייספר כהצלחה. הפוך: תשובה כמו "The answer is the well-known Peter Principle" תתאים ל"Peter" אבל
   גם ל"Jones" אם המילה מופיעה בהקשר שונה. זה מאוד fragile ומשפיע על כל הניסויים.

  המלצה: להשתמש ב-exact match או fuzzy match עם threshold, ולהוסיף validation ידני על מדגם.

  4. Activation Patching — אורך רצף שונה

  experiments/06_mechanistic_analysis/activation_patching.py:122:
  clean_h = clean_hidden_states[patch_layer][0, -1, :].clone()

  ה-clean hidden states מגיעים מ-neutral prompt (קצר) וה-patching נעשה ב-sycophantic prompt (ארוך). הפוזיציה -1 היא
  token שונה לגמרי בשני הרצפים. מושגית זה עדיין אומר "ה-representation במקום הניבוי", אבל positional encoding שונה
  לגמרי, מה שעלול להחליש את התוצאות או להכניס רעש.

  ---
  בעיות מתודולוגיות

  5. Cosine similarity ~0.05 אינו ראיה לאורתוגונליות

  ניסוי 05 טוען ש-cosine ~0.05 מראה שכיווני שקר שונים הם "orthogonal". אבל במרחב 4096-ממדי, וקטורים אקראיים נוטים להיות
  כמעט אורתוגונליים. צריך baseline: מה ה-cosine הצפוי בין שני וקטורים אקראיים באותו ממד? (בערך 1/√d ≈ 0.016). אם 0.05 לא
   שונה מהציפייה הזו, זו לא ממצא — זו סטטיסטיקה של מרחבים ממדיים גבוהים.

  תיקון: להוסיף permutation baseline ל-cosine similarity.

  6. Cross-model Procrustes alignment עם supervision

  experiments/04_cross_model_transfer/run.py:257-258:
  R, _ = orthogonal_procrustes(X_tgt_s[:min(min_src, min_tgt)],
                                X_src_s[:min(min_src, min_tgt)])

  ה-Procrustes alignment נעשה על דאטה מתויג (חצי ראשון = lied, חצי שני = resisted). זה נותן ל-alignment יתרון כי הוא
  רואה את מבנה הקלאסים. fair transfer test צריך alignment על דאטה ללא תיוג, או דאטה עם תיוג אחר.

  7. Cross-question generalization — מטריקה שגויה

  experiments/02_confound_free_detection/run.py:195:
  cross_q[layer] = float(clf.score(scaler.transform(X_test), y_test))

  clf.score() מחזיר accuracy רגיל, לא balanced accuracy. כל שאר הקוד משתמש ב-balanced accuracy. חוסר עקביות.

  ---
  בעיות קוד קטנות

  8. קריסה שקטה ב-Exp 06a

  experiments/06_mechanistic_analysis/logit_lens.py:325-326:
  "median_flip_layer": median_flip if lie_trajectories and flip_layers else None,

  אם lie_trajectories לא ריק אבל flip_layers ריק (כל ה-flip_layer הם None), median_flip ו-mean_flip לא מוגדרים ותהיה
  NameError.

  9. Dataset parsing fragile

  src/utils.py:99: i += 4 — hardcoded step שמניח מבנה מאוד ספציפי של הדאטאסט. אם שורה אחת חסרה, כל ה-alignment נשבר. אין
   validation.

  10. קוד מת/כפול

  experiments/04_cross_model_transfer/run.py:270:
  clf.fit(X_src_s if HAS_SCIPY else X_src_s, y_src)

  X_src_s if HAS_SCIPY else X_src_s — תמיד אותו ערך. כנראה צריך להיות X_src_s במקרה אחד ו-PCA-transformed במקרה השני.

  ---
  המלצות לשיפור (לקראת פרסום)

  1. תקנו את ה-data leakage — זה הדבר הכי חשוב. עם Pipeline, התוצאות עלולות לרדת קצת, אבל יהיו אמיתיות.
  2. הוסיפו random baseline ל-cosine similarity — חשוב לטענה "no single lie direction".
  3. שפרו את answer matching — fuzzy match עם threshold + ולידציה ידנית על מדגם.
  4. הוסיפו probe על hidden states מה-prompt (לפני generation) — אם אפשר לזהות "כוונה לשקר" לפני שהמודל מתחיל לייצר, זה
  ממצא הרבה יותר חזק.
  5. הוסיפו visualization — plots של ה-logit lens trajectories ואת attention patterns ישדרגו מאוד את המאמר.
  6. README: ה-clone URL אומר Maor36/deception-probe אבל הריפו נמצא ב-cohen-liel/deception-probe.

  ---
  סיכום

  המחקר עצמו מצוין מבחינה רעיונית — ה-same-prompt design וההפרדה lie/hallucination הם תרומות משמעותיות לתחום. הקוד נקי
  ומאורגן. אבל ה-data leakage ב-probing ו-ה-answer matching החלש הם באגים שחייבים לתקן לפני submission, כי reviewer טוב
  יתפוס אותם. לאחר התיקונים, אם התוצאות מחזיקות, זה מאמר שיכול להתקבל בכנס מוביל.
