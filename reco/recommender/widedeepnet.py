import pandas as pd
from keras.layers import Input, Dense, Embedding, Lambda, Add, concatenate, Reshape
from keras.models import Model

class WideAndDeepNetwork:

    def __init__(self,
                 deep_embeddings,
                 deep_layers,
                 n_factors = 5
                 ):
        self.embeddings = deep_embeddings
        self.deep_layers = deep_layers
        self.n_factors = n_factors

    def Val2IndexFit(self, df, cols):
        """
        :param df: input data
        :param cols: the columns in the data which needs to be transformed
        :return:
        """
        colUniq = {}
        uniqDict = {}
        for col in cols:
            colUniq[col] = df[col].unique()

        for key, val in colUniq.items():
            uniqDict[key] = {v:i for i,v in enumerate(colUniq[key])}
            uniqDict[key]['element_not_existent'] = len(colUniq[key])

        self.uniqDict = uniqDict

    def Val2IndexTransform(self, df):
        for key,val in self.uniqDict.items():
            df[key] = df[key].apply(lambda x: val[x] if x in val else val['element_not_existent'])

        return df

    # create dense embeddings from sparse categorical features in deep part
    def EmbeddingInput(self, inp, n_in, n_out):
        out = Embedding(input_dim=n_in, output_dim=n_out, input_length=1)(inp)
        return out

    # create one-hot representations for the categorical features in wide part
    def OneHotInput(self, inp, n_classes):
        out = Embedding(input_dim=n_classes, output_dim=n_classes, embeddings_initializer='identity',
                      trainable=False, input_length=1)(inp)
        return out

    # handle all numerical features for both deep and wide parts 
    def NumericalInput(self, inp):
        inp = Reshape((1, 1))(inp)
        out = Lambda(lambda x: x)(inp)
        return out

    def createModel(self, X, y, categorical_cols):
        """
        Args:
            X: input data
            y: input train ratings
            categorical_cols: the categorical columns in the data

        Returns: the final model
        """
        inputLayer = []
        for col in X.columns:
            if col not in categorical_cols:
                inp = Input((1,), dtype='float32')
            else:
                inp = Input((1,), dtype='int64')
            inputLayer.append(inp)

        deepNetLayer = self.deep(X, y, inputLayer, categorical_cols)
        wideNetLayer = self.wide(X, y, inputLayer, categorical_cols)

        outputLayer = Add()([deepNetLayer, wideNetLayer])
        model = Model(inputs=inputLayer, outputs=outputLayer)
        return model


    def deep(self, X, y, inputLayer, categorical_cols):

        """
        the deep part of the model. The input has two parts - either categorical inputs
        which converts sparse features to dense representations and numerical inputs

        :param X: input data
        :param y: input labels
        :param categorical_cols: the columns in X that needs to be embedded i.e. all non numerical columns
        :return: the output layer from the deep part
        """

        firstLayer = []
        for i, col in enumerate(X.columns):
            inp = inputLayer[i]
            if col in categorical_cols:
                embed_out = self.EmbeddingInput(inp, len(self.uniqDict[col]), self.n_factors)
                firstLayer.append(embed_out)
            else:
                num_out = self.NumericalInput(inp)
                firstLayer.append(num_out)

        firstDimension = self.n_factors * len(categorical_cols) + len(X.columns) - len(categorical_cols)
        print("first dimension {}".format(firstDimension))

        currentLayer = concatenate(firstLayer)
        currentLayer = Reshape(target_shape=(firstDimension,))(currentLayer)
        for layer_units in self.deep_layers:
            currentLayer = Dense(units=layer_units)(currentLayer)

        outputLayer = Dense(units=1)(currentLayer)
        return outputLayer

    def wide(self, X, y, inputLayer, categorical_cols):
        """
        The wide part of the model. This practically serves as a linear regressor.

        :param X: input data
        :param y: input ratings
        :param categorical_cols: the features that need to be one-hot encoded
        :return: the output layer from the wide part
        """

        # currently NOT supporting crossed columns

        firstLayer = []
        dims = 0
        for i, col in enumerate(X.columns):
            inp = inputLayer[i]
            if col in categorical_cols:
                embed_out = self.OneHotInput(inp, len(self.uniqDict[col]))
                firstLayer.append(embed_out)
                dims += len(self.uniqDict[col])
            else:
                num_out = self.NumericalInput(inp)
                firstLayer.append(num_out)
                dims += 1

        #print("first dimension {}".format(dims))

        firstLayer = concatenate(firstLayer)
        firstLayer = Reshape(target_shape=(dims,))(firstLayer)
        outputLayer = Dense(units=1)(firstLayer)
        return outputLayer


    def fit(self, X, y, categorical_cols):
        """
        Args:
            X: input data
            y: input ratings
            categorical_cols: the categorical features that need to be embedded or one-hot encoded

        Returns: None
        """
        self.Val2IndexFit(X, categorical_cols)
        X = self.Val2IndexTransform(X)
        self.wdModel = self.createModel(X, y, categorical_cols)

        from keras.optimizers import Adam
        a = Adam(lr=0.01)

        self.wdModel.compile(loss='mean_squared_error', optimizer=a)

        from sklearn.model_selection import train_test_split

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

        X_train = [X_train[col] for col in X.columns]
        X_valid = [X_valid[col] for col in X.columns]
        history = self.wdModel.fit(x=X_train, y=y_train, validation_data=[X_valid, y_valid], epochs=20, batch_size=32, verbose=2)

    def predict(self, X):
        X = self.Val2IndexTransform(X)
        XX = [X[col] for col in X.columns]
        return self.wdModel.predict(x=XX)
