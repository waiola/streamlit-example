import streamlit as st
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('vineyard_weather_1948-2017.csv')

# Convert date to datetime format and extract the week number and year
data['DATE'] = pd.to_datetime(data['DATE'])
data['Year'] = data['DATE'].dt.year
data['Week'] = data['DATE'].dt.isocalendar().week

# Filter data for weeks 35 to 40
filtered_data = data[(data['Week'] >= 35) & (data['Week'] <= 40)]

# Aggregate data by Year and Week
weekly_data = filtered_data.groupby(['Year', 'Week']).agg({
    'PRCP': 'sum',  # Total weekly precipitation
    'TMAX': 'max'   # Maximum temperature of the week
}).reset_index()

# Create features for weeks 35 to 39 and target for week 40
features = weekly_data.pivot(index='Year', columns='Week', values=['PRCP', 'TMAX'])
features.columns = ['_'.join(map(str, col)).strip() for col in features.columns.values]
features = features.dropna(subset=['PRCP_40', 'TMAX_40'])

# Define the storm condition for week 40
features['Storm_40'] = ((features['PRCP_40'] >= 0.35) & (features['TMAX_40'] <= 80)).astype(int)

# Prepare features and target
X = features.drop(columns=['PRCP_40', 'TMAX_40', 'Storm_40'])
y = features['Storm_40']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Compute model metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Extract true positives, true negatives, false positives, and false negatives
TP = conf_matrix[1][1]
TN = conf_matrix[0][0]
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]

# Compute sensitivity and specificity
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)


# Assume sensitivity and specificity are calculated from the model elsewhere and imported here
sensitivity = 0.75  # Placeholder value
specificity = 0.85  # Placeholder value

# Calculate dependent probabilities
def calculate_values(p_noble_rot, p_no_sugar, p_typical_sugar, p_high_sugar):
    p_dns = 0.5 * specificity + 0.5 * (1 - sensitivity)
    p_ds = 1 - p_dns

    p_ns_dns = 0.5 * specificity / (0.5 * specificity + 0.5 * (1 - sensitivity))
    p_s_ds = 0.5 * sensitivity / (0.5 * sensitivity + 0.5 * (1 - specificity))

    e_no_storm = p_no_sugar * 960000 + p_typical_sugar * 1410000 + p_high_sugar * 1500000
    e_storm = p_noble_rot * 3300000 + (1 - p_noble_rot) * 420000

    e_nh_dns = p_ns_dns * e_no_storm + (1 - p_ns_dns) * e_storm
    e_nh = p_dns * e_nh_dns + p_ds * (p_s_ds * e_storm + (1 - p_s_ds) * e_no_storm)
    
    value_of_data = e_nh - 960000
    return value_of_data, e_nh, p_ds * (p_s_ds * e_storm + (1 - p_s_ds) * e_no_storm)

# Streamlit interface
st.title("Vineyard Decision Assistant")
p_noble_rot = st.slider('Chance of Noble Rot (P(noble rot))', 0.0, 1.0, 0.5)
p_no_sugar = st.slider('Probability of No Sugar', 0.0, 1.0, 0.1)
p_typical_sugar = st.slider('Probability of Typical Sugar', 0.0, 1.0, 0.3)
p_high_sugar = st.slider('Probability of High Sugar', 0.0, 1.0, 0.6)

if st.button('Calculate Economic Values'):
    value_of_data, e_nh, e_h_ds = calculate_values(p_noble_rot, p_no_sugar, p_typical_sugar, p_high_sugar)
    st.write(f"E(H|DS): {e_h_ds}")
    st.write(f"E(NH): {e_nh}")
    st.write(f"Value of Data: {value_of_data}")

    if value_of_data > 0:
        st.success("Recommend buying the detector and making decision based on the detector.")
    elif value_of_data == 0:
        st.info("There is no difference between choices.")
    else:
        st.error("Recommend harvest now.")
