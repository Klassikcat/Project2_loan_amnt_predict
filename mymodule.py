import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import plot_confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, SelectKBest

def EDA_chi2(df, validation, test, delete, target): #EDA Process
    import pandas as pd
    dels = [col for col in df.columns if (delete in col)]
    df.drop(columns=dels, inplace=True)
    categories = df.drop(target, axis=1).columns
    features = []
    for i in categories:
        ex = pd.concat([pd.DataFrame(df[i]), df[target]], axis=1)#make tidy data
        cont = pd.crosstab(df[i], ex[target])
        chirel = stats.chi2_contingency(cont, correction=False) #test dependency of specific feature and target
        if chirel[1] > 0.05: #chirelp[1] == pvalue. if pvalue > 0.05: two features are independent. else: two features are dependent.
            continue
        elif chirel[1] < 0.05:
            print(f"feature name: {i}, correlationship: {chirel[0]}, pvalue:{chirel[1]}")
            features.append(i) #add dependent features
    print(f"\nsize of original features: {len(df.columns)}\nsize of features: {len(features)}\nlist of features: {features}")
    df_x = df[features]
    df_y = df[target]
    validation_x = validation[features]
    validation_y = validation[target]
    test = test[features]
    return df_x, df_y, validation_x, validation_y, test


def f1beta_score_py(TP, TN, FP, FN, beta):
    total = TP + TN + FP + FN
    acc = TP + TN / total
    prs = TP / (TP + FP)
    rec = TP / (TP + FN)
    return (1+beta**2) * (prs*rec/((beta**2 * prs) + rec))

def precision(matrix):
    TP, TN, FP, FN = matrix[1, 1], matrix[0, 0], matrix[0, 1], matrix[1, 0]
    total = TP + TN + FP + FN
    acc = (TP + TN) / total
    prs = TP / (TP + FP)
    rec = TP / (TP + FN)
    return acc, prs, rec

def confusionmatrix(pipe, X_val, y_val):
    fig, ax = plt.subplots()
    pcm = plot_confusion_matrix(pipe, X_val, y_val,
                            cmap=plt.cm.Blues,
                            ax=ax);
    plt.title(f'Confusion matrix, n = {len(y_val)}', fontsize=15)
    plt.show()
    
def toint(string): #make sheet datas to floating number
  if type(string) == str:
    return int(string)
  else:
    return int(string)

def selectfeatures(train_x, train_y, vali_x, vali_y, model, ordinal, categorical):
    selected_columns = []
    mae_score_list = []
    r2_score_list = []
    rmse_score_list = []
    encoding = make_pipeline(OrdinalEncoder(cols=ordinal, handle_unknown='ignore'), OneHotEncoder(cols=categorical, handle_unknown='ignore'))
    encoded_train = encoding.fit_transform(train_x, train_y)
    encoded_vali = encoding.transform(vali_x)
    for k in range(1, len(encoded_train.columns)+1):
        selector = SelectKBest(score_func=f_regression, k=k)
        train_x_selected = selector.fit_transform(encoded_train, train_y)
        vali_x_selected = selector.transform(encoded_vali)
        all_names = encoded_train.columns
        selected_mask = selector.get_support()
        selected_names = all_names[selected_mask]
        
        model = model
        model.fit(train_x_selected, train_y)
        pred_y = model.predict(vali_x_selected)
        mae = mean_absolute_error(vali_y, pred_y)
        rmse = np.sqrt(mean_squared_error(vali_y, pred_y))
        r2 = r2_score(vali_y, pred_y)
        
        selected_columns.append(selected_names.tolist())
        r2_score_list.append(r2)
        mae_score_list.append(mae)
        rmse_score_list.append(rmse)

    con = pd.DataFrame([r2_score_list, mae_score_list, rmse_score_list, selected_columns], index=['r_squared', 'mae', 'rmse', 'selected_columns']).transpose()
    return con