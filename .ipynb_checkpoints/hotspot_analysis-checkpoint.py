import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import folium

def run_hotspot_analysis():
    data_path = 'C:/Users/harsh/Downloads/flask_app/20_Victims_of_rape.csv'

    coords_path = 'C:/Users/harsh/Downloads/flask_app/India States-UTs.csv'

    data = pd.read_csv(data_path)
    coords_data = pd.read_csv(coords_path)

    # Normalize the names
    data['Area_Name'] = data['Area_Name'].str.lower().str.strip()
    coords_data['State/UT'] = coords_data['State/UT'].str.lower().str.strip()

    # Merge the datasets
    data_merged = pd.merge(data, coords_data, left_on='Area_Name', right_on='State/UT', how='left')
    data_india = data_merged.dropna().copy()

    # Encode categorical features
    label_encoder_area = LabelEncoder()
    label_encoder_subgroup = LabelEncoder()
    data_india['Area_Name_Encoded'] = label_encoder_area.fit_transform(data_india['Area_Name'])
    data_india['Subgroup_Encoded'] = label_encoder_subgroup.fit_transform(data_india['Subgroup'])

    # Features and target
    X = data_india[['Area_Name_Encoded', 'Year', 'Subgroup_Encoded']]
    y = data_india['Rape_Cases_Reported']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict and classify crime severity
    data_india['Predicted_Cases'] = rf_model.predict(X)
    data_india['Severity_Score'] = 10 * (data_india['Rape_Cases_Reported'] - data_india['Rape_Cases_Reported'].min()) / (data_india['Rape_Cases_Reported'].max() - data_india['Rape_Cases_Reported'].min())

    def get_color(score):
        if score >= 8:
            return 'red'
        elif score >= 6:
            return 'orange'
        elif score >= 4:
            return 'yellow'
        elif score >= 2:
            return 'blue'
        else:
            return 'green'

    # Create the map
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    for _, row in data_india.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['Rape_Cases_Reported'] / 50,
            color=get_color(row['Severity_Score']),
            fill=True,
            fill_color=get_color(row['Severity_Score']),
            fill_opacity=0.1,
            tooltip=f"Area: {row['Area_Name']}<br>Crime Severity: {row['Severity_Score']:.1f}/10"
        ).add_to(m)

    return m._repr_html_()
