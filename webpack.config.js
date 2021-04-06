const path = require('path');

module.exports = {
    entry: './dist/knn-recommender.js',
    output: {
        path: path.resolve(__dirname, 'distfornpmpublishing'),
        filename: 'knn-recommender.js',
        globalObject: 'this',
        library: {
            name: 'knn-recommender',
            type: 'umd',
        }
    },
};