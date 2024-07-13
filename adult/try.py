from ucimlrepo import fetch_ucirepo

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets
print(X)
print(y)
for index, r in X.iterrows():
    if index == 0:
        sent = (
            f"age: {r['age']}, workclass: {r['workclass']}, fnlwgt: {r['fnlwgt']}, education: {r['education']}, "
            f"education-num: {r['education-num']}, marital-status: {r['marital-status']}, occupation: {r['occupation']}, "
            f"relationship: {r['relationship']}, race: {r['race']}, capital-gain: {r['capital-gain']}, "
            f"capital-loss: {r['capital-loss']}, hours-per-week: {r['hours-per-week']}, "
            f"native-country: {r['native-country']}"
        )
        print(sent)