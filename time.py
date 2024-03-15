def plot_confusion_matrix(y_train, y_test, labels):
    """
    Function to plot confusion matrix.
    """
    cm = confusion_matrix(y_train, y_test)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot()

labels=['Age','JobLevel','MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','Performance']
plot_confusion_matrix(y_train, y_test, labels)