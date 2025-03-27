
# Load dataset (replace with actual dataset)
data = pd.read_csv('intrusion_data.csv')
X = data.drop(columns=['label'])  # Features
y = data['label']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for KNN and Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize individual models
dt = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
lgbm = lgb.LGBMClassifier()

# Define Neural Network model
class NeuralNetworkWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.input_dim,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def fit(self, X, y):
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        return self
    
    def predict_proba(self, X):
        return np.hstack([1 - self.model.predict(X), self.model.predict(X)])
    
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

nn = NeuralNetworkWrapper(input_dim=X_train.shape[1])

# Create ensemble model with soft voting
ensemble = VotingClassifier(estimators=[
    ('dt', dt),
    ('knn', knn),
    ('lgbm', lgbm),
    ('nn', nn)
], voting='soft')

# Train ensemble model
ensemble.fit(X_train_scaled, y_train)

# Evaluate model
accuracy = ensemble.score(X_test_scaled, y_test)
print(f'Ensemble Model Accuracy: {accuracy * 100:.2f}%')
