@st.cache_resource
def train_model():

    df = pd.read_csv("fixed_dataset.csv")
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Order Value": "Order_Value",
        "Plant_Load %": "Plant_Load_%"
    })

    features = [
        "Quantity",
        "Ship Quantity",
        "Order_Value",
        "Days_Left",
        "Order_Age",
        "PR_Delay_Days",
        "Plant_Load_%",
        "Not_Shipped_Flag"
    ]

    X = df[features]
    y = df["Delayed"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,              # Prevent overfitting
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
