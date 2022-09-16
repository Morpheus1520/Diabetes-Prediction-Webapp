import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from PIL import Image

diabetes_dataset = pd.read_csv(r"diabetes.csv")
diabetes_dataset_preprocessed = pd.read_csv(r"diabetes_preprocessed.csv")
diabetes_dataset_preprocessed = diabetes_dataset_preprocessed.drop(columns=["Unnamed: 0"])

loaded_model = pickle.load(open(r"votingclf_trained.sav", "rb"))

scaler = pickle.load(
    open(r"std_scaler.pkl", "rb"))


def diabetes_prediction(input_data):
    # defining Indices
    pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age = 0, 1, 2, 3, 4, 5, 6, 7

    N1 = [1 if input_data[age] <= 30 and input_data[glucose] <= 120 else 0]

    N2 = [1 if input_data[bmi] <= 30 else 0]

    N3 = [1 if input_data[age] <= 30 and input_data[pregnancies] <= 6 else 0]

    N4 = [1 if input_data[bloodpressure] <= 80 and input_data[glucose] <= 105 else 0]

    N5 = [1 if input_data[skinthickness] <= 20 else 0]

    N6 = [1 if input_data[bmi] < 30 and input_data[skinthickness] <= 20 else 0]

    N7 = [1 if input_data[bmi] <= 30 and input_data[glucose] <= 105 else 0]

    N9 = [1 if input_data[insulin] <= 200 else 0]

    N10 = [1 if input_data[bloodpressure] < 80 else 0]

    N11 = [1 if input_data[pregnancies] < 4 and input_data[pregnancies] != 0 else 0]

    N12 = [1 if input_data[age] <= 30 and input_data[bloodpressure] <= 90 else 0]

    input_data_as_nparray = np.asarray(input_data)
    input_data_reshaped = input_data_as_nparray.reshape(1, -1)

    input_data_scaled = scaler.transform(input_data_reshaped)

    final_input_data = np.append(input_data_scaled,
                                 [N1[0], N2[0], N3[0], N4[0], N5[0], N6[0], N7[0], N9[0], N10[0], N11[0], N12[0]])

    prediction = loaded_model.predict(final_input_data.reshape(1, -1))

    if prediction == 0:
        return "Person is Not Diabetic"
    elif prediction == 1:
        return "Person is Diabetic"


def main():
    with st.sidebar:
        selected = option_menu("Main Menu", ["Read", "Predict"], icons=["journal-bookmark-fill", "code-slash"])

    if selected == "Read":

        st.title("About this project: Diabetes in a nutshell")

        st.subheader("This section will help you broaden your horizons")

        st.write("""Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into 
        energy. 

Your body breaks down most of the food you eat into sugar (glucose) and releases it into your bloodstream. When your 
blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let the blood sugar into 
your body’s cells for use as energy. 

With diabetes, your body doesn’t make enough insulin or can’t use it as well as it should. When there isn’t enough 
insulin or cells stop responding to insulin, too much blood sugar stays in your bloodstream. Over time, 
that can cause serious health problems, such as heart disease, vision loss, and kidney disease. 

There isn’t a cure yet for diabetes, but losing weight, eating healthy food, and being active can really help.""")

        st.subheader("What are the different types of diabetes?")

        st.write("""The most common types of diabetes are type 1, type 2, 
        and gestational diabetes. 

Type 1 diabetes If you have type 1 diabetes, your body does not make insulin. Your immune system attacks and destroys 
the cells in your pancreas that make insulin. Type 1 diabetes is usually diagnosed in children and young adults, 
although it can appear at any age. People with type 1 diabetes need to take insulin every day to stay alive. 

Type 2 diabetes If you have type 2 diabetes, your body does not make or use insulin well. You can develop type 2 
diabetes at any age, even during childhood. However, this type of diabetes occurs most often in middle-aged and older 
people. Type 2 is the most common type of diabetes. 

Gestational diabetes Gestational diabetes develops in some women when they are pregnant. Most of the time, 
this type of diabetes goes away after the baby is born. However, if you’ve had gestational diabetes, you have a 
greater chance of developing type 2 diabetes later in life. Sometimes diabetes diagnosed during pregnancy is actually 
type 2 diabetes. 

Other types of diabetes Less common types include monogenic diabetes, which is an inherited form of diabetes, 
and cystic fibrosis-related diabetes. """)

        st.subheader("Now lets talk about our data")

        st.table(diabetes_dataset.head(10))

        st.markdown("**Target Distribution**")

        st.write(
            "The graph below shows that the data is unbalanced. The number of diabetic patients is 268 the number of "
            "non-diabetic patients is 500")

        target_distribution_img = Image.open(r"plots/distribution_by_target.png")
        st.image(target_distribution_img)

        st.markdown("**Missing Values**")

        st.write(
            "It is quite noticeable that our data also has a lot of missing values. Some features such as Blood "
            "Pressure, Skin Thickness, Insulin, BMI, etc have values 0 in some places which don't make sense, "
            "hence it is safe to treat these values as missing.")

        diabetes_dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = diabetes_dataset[
            ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)

        st.code(
            """diabetes_dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = diabetes_dataset[[
            "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)""",
            "python")

        missing_img = Image.open(r"plots/missing_values.png")
        st.image(missing_img)

        st.markdown("**Imputing the Missing Values**")

        st.write("Since the data is MCAR (Missing Completely At Random) we'll go with median imputation strategy")

        st.write(diabetes_dataset.groupby("Outcome").median())

        st.caption("Median values per feature of the Data grouped by 'Outcome'.")

        st.write("We have now imputed the missing values with respect to the records in the table above")

        imputed_dataset = Image.open(r"plots/imputed_dataset.png")
        st.image(imputed_dataset)

        st.subheader("Feature creation and Exploratory data analysis (EDA)")

        st.write(
            "Now we'll create new features with the help of graphical representation of our data. The scatterplots "
            "shown below will help us visualize the concentration of healthy and diabetic people with respect to "
            "distinct features.")

        st.markdown("**N1 (Age <= 30 and Glucose <= 120)**")

        n1scatter = Image.open(r"plots/n1scatter.png")
        st.image(n1scatter)

        st.write(
            "From the plot above we can observe that healthy people are concentrated between age <= 30 and glucose <= "
            "120. "
            "Hence, our new feature **N1** will be 1 where age <= 30 and glucose <= 120, 0 otherwise.")

        st.code("""diabetes_dataset["N1"] = 0
diabetes_dataset.loc[(diabetes_dataset["Glucose"] <= 120) & (diabetes_dataset["Age"] <= 30), "N1"] = 1""", "python")

        st.write(
            "Now, out of the instances in N1 we find out how many of them are diabetic and how many are non-diabetic")

        n1bar = Image.open(r"plots/n1bar.png")
        st.image(n1bar)

        st.write("""Out of all the 268 Diabetic people, 242 fall under N1=0 and only 26 fall under N1=1 <br> 
        Similarly, Out of all the 500 Healthy people, 256 fall under N1=0 and 244 fall under N1=1 <br> This signifies 
        that majority of the people in N1 are healthy.""",
                 unsafe_allow_html=True)

        n1pie = Image.open(r"plots/n1pie.png")
        st.image(n1pie, caption="""N1 distribution by target (Age <= 30 and Glucose <= 120)""")

        st.write("")
        st.write("Similarly, we create more such features.")

        st.markdown("**N2 (BMI <= 30)**")

        st.markdown("""**BMI**: According to wikipedia "The body mass index (BMI) or Quetelet index is a value 
        derived from the mass (weight) and height of an individual. The BMI is defined as the body mass divided by 
        the square of the body height, and is universally expressed in units of kg/m2, resulting from mass in 
        kilograms and height in metres. <br> <br> 30 kg/m² is the limit to obesity.""", unsafe_allow_html=True)

        n2bar = Image.open(r"plots/bmibar.png")
        st.image(n2bar)

        n2pie = Image.open(r"plots/bmipie.png")
        st.image(n2pie, "N2 distribution by target")

        st.markdown("**N3 (pregnancies <= 6 and Age <= 30)**")

        n3scatter = Image.open(r"plots/n3scatter.png")
        st.image(n3scatter)

        st.write(
            "From the scatterplot above notice that majority of the Healthy people are concentrated between age <= 30 "
            "and pregnancies <= 6")

        n3bar = Image.open(r"plots/n3bar.png")
        st.image(n3bar)

        st.write("""This Piechart signifies that out of all the 268 diabetic people, 32.1% fell under N3 and 67.9% 
        fell under ROW. <br> Similarly out of all the 500 Healthy people 65% fell under N3.""", unsafe_allow_html=True)

        n3pie = Image.open(r"plots/n3pie.png")
        st.image(n3pie, "N3 distribution by target")

        st.markdown("**N4 (BloodPressure <= 80 and Glucose <= 105)**")

        n4scatter = Image.open(r"plots/n4scatter.png")
        st.image(n4scatter)

        st.write(
            "From the scatterplot above notice that majority of the Healthy people are concentrated between "
            "BloodPressure <= 80 and Glucose <= 105")

        n4bar = Image.open(r"plots/n4bar.png")
        st.image(n4bar)

        n4pie = Image.open(r"plots/n4pie.png")
        st.image(n4pie, "N4 distribution by target")

        st.markdown("**N5 (SkinThickness <= 20)**")

        n5bar = Image.open(r"plots/n5bar.png")
        st.image(n5bar)

        n5pie = Image.open(r"plots/n5pie.png")
        st.image(n5pie, "N5 distribution by target")

        st.markdown("**N6 (SkinThickness <= 20 and BMI < 30)**")

        n6scatter = Image.open(r"plots/n6scatter.png")
        st.image(n6scatter)

        st.write(
            "From the scatterplot above notice that majority of the Healthy people are concentrated between "
            "SkinThickness <= 20 and BMI < 30")

        n6bar = Image.open(r"plots/n6bar.png")
        st.image(n6bar)

        n6pie = Image.open(r"plots/n6pie.png")
        st.image(n6pie, "N6 distribution by target")

        st.markdown("**N7 (Glucose <= 105 and BMI <= 30)**")

        n7scatter = Image.open(r"plots/n7scatter.png")
        st.image(n7scatter)

        st.write(
            "From the scatterplot above notice that majority of the Healthy people are concentrated between Glucose "
            "<= 105 and BMI <= 30")

        n7bar = Image.open(r"plots/n7bar.png")
        st.image(n7bar)

        n7pie = Image.open(r"plots/n7pie.png")
        st.image(n7pie, "N7 distribution by target")

        st.markdown("**N9 (Insulin <= 200)**")

        n9bar = Image.open(r"plots/n9bar.png")
        st.image(n9bar)

        n9pie = Image.open(r"plots/n9pie.png")
        st.image(n9pie, "N9 distribution by target")

        st.markdown("**N10 (BloodPressure < 80)**")

        n10bar = Image.open(r"plots/n10bar.png")
        st.image(n10bar)

        n10pie = Image.open(r"plots/n10pie.png")
        st.image(n10pie, "N10 distribution by target")

        st.markdown("**N11 (Pregnancies between 1 - 3)**")

        n11bar = Image.open(r"plots/n11bar.png")
        st.image(n11bar)

        n11pie = Image.open(r"plots/n11pie.png")
        st.image(n11pie, "N11 distribution by target")

        st.markdown("**N12 (Age <= 30 and BloodPressure <= 90)**")

        n12scatter = Image.open(r"plots/n12scatter.png")
        st.image(n12scatter)

        st.write(
            "From the scatterplot above notice that majority of the Healthy people are concentrated between Age <= 30 "
            "and BloodPressure <= 90")

        n12bar = Image.open(r"plots/n12bar.png")
        st.image(n12bar)

        n12pie = Image.open(r"plots/n12pie.png")
        st.image(n12pie, "N12 distribution by target")

        st.subheader("Preparation of Dataset")

        st.write(
            "Before feeding our data to the Machine Learning model, it is crucial to standardize the feature vector ("
            "X). Standardization makes all variables contribute equally to the model.")

        st.code("""from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled = scaler.fit_transform(num_df)
num_df = pd.DataFrame(scaled, columns=num_df.columns)
num_df.head()""", "python")

        scaled_df = Image.open(r"plots/scaled_df.png")
        st.image(scaled_df)

        st.code("""final_df = pd.concat([num_df, binary_df], axis=1)""", "python")

        st.write("This is our final dataset with existing and created features.")
        st.table(diabetes_dataset_preprocessed.head())

        corr = Image.open(r"plots/correlation_heatmap.png")
        st.image(corr, "Correlation Heatmap")

        st.code("""X = final_df.drop(columns="Outcome", axis=1)
y = final_df["Outcome"]""", "python")

        st.subheader("Machine Learning")

        st.write("""LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is 
        designed to be distributed and efficient with the following advantages:""")

        st.markdown("- Faster training speed and higher efficiency.")
        st.markdown("- Lower memory usage.")
        st.markdown("- Better Accuracy.")
        st.markdown("- Support of parallel and GPU learning.")
        st.markdown("- Capable of handling large-scale data.")

        st.write("""To find the best hyperparameters, we'll use Random Search CV.

Random search is a technique where random combinations of the hyperparameters are used to find the best solution for 
the built model. Generally RS is more faster and accurate than GridSearchCV who calculate all possible combinations. 
With Random Grid we specify the number of combinations that we want""")

        st.write("""LightGBM splits the tree leaf-wise as opposed to other boosting algorithms that grow tree 
        level-wise. It chooses the leaf with maximum delta loss to grow. Since the leaf is fixed, the leaf-wise 
        algorithm has lower loss compared to the level-wise algorithm.""")

        lgbm = Image.open(r"plots/lgbm.png")
        st.image(lgbm)

        st.write(
            "To improve the accuracy even more, we can add a KNeighborsClassifier to Lightgbm (Voting Classifier).")
        st.markdown("""**KNeighborsClassifier** : KNeighborsClassifier implements learning based on the k nearest 
        neighbors of each query point, where k is an integer value specified by the user. 

**VotingClassifier** : VotingClassifier is a meta-classifier for combining similar or conceptually different machine 
learning classifiers for classification via majority or plurality voting""")

        st.write("The best parameters have been selected after running GridSearchCV and RandomizedSearchCV in the "
                 "background.")

        st.code("""knn_clf = KNeighborsClassifier(n_neighbors = 23)

voting_clf = VotingClassifier (
        estimators = [('knn', knn_clf), ('lgbm', lgbm_clf)],
                     voting='soft', weights = [1,1])
voting_clf.fit(X_train, y_train)""", "python")

        st.markdown("**Our model is now trained with an accuracy of 88.31%**")

    elif selected == "Predict":

        st.title("Diabetes Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.number_input("Number of Pregnancies")

        with col2:
            glucose = st.number_input("Glucose level")

        with col3:
            bloodpressure = st.number_input("Blood Pressure")

        with col1:
            skinthickness = st.number_input("Skin Thickness")

        with col2:
            insulin = st.number_input("Insulin level")

        with col3:
            bmi = st.number_input("Body Mass Index")

        with col1:
            dpf = st.number_input("Diabetes Pedigree Function")

        with col2:
            age = st.number_input("Age")

        # Prediction
        final_pred = ""

        # Create a button
        if st.button("Diabetes Prediction Result"):
            final_pred = diabetes_prediction(
                [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age])

        st.success(final_pred)


if __name__ == '__main__':
    main()
