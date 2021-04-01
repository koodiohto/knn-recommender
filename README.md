# knn-recommender
Pure JavaScript implementation of a K-nearest neighbour based collaborative filtering recommender for like/dislike user item matrices. This library should run in node and browser environments. This is an exprimental implementation. If you are looking for a more high performing library, I'd recommend you to check out [recommendationRacoon](https://github.com/guymorita/recommendationRaccoon).

The recommender takes a user item matrix of size X x Y where X[0] column represents the user id's and Y[0] the item labels. The cells in the matrix are expected to contain either -1 (dislike), 0 (no rating given) or 1 (like). This information can be used to calculate the similarity of users in the matrix based on jaccard similarity.

The Jaccard similarity calculates the common ratings between two users and divides that by the total ratings given by the users. The non-ratings are not considered when calculating the similarity of two users.

So if user X has a rating matrix (1, -1, 0, 0, -1, 1, 0) and Y (1, -1, 0, 1, -1, 0, 0) their Jaccard similarity is (1+1+1) / 5 = 3/5. 5 is effectively the number of the elements that at least one of the two users has either liked or disliked. Based on this we could provide a recommendation for user Y for item number 6 as Y hasn't expressed any preference for that and X has liked this item.

This recommender can also work only based on non recommendations and recommendations (0's and 1's so it's not necessary to provide dislikes (-1).

# Installation
```bash
npm install --save knn-recommender
```

# Basic usage

```js
const kNNRecommender = new KNNRecommender([['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4','item 5', 'item 6', 'item 7'], ['user 1', 1, -1, 0, 0, -1, 1, 0], ['user 2', 1, -1, 0, 1, -1, 0, 0]])
kNNRecommender.initializeRecommender().then(() => {
    const userRecommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 3')
    console.log(`new recommendation for user 3 ${userRecommendations[0].itemId}`)
})
```

# API

Method             | Arguments          | Returns      | Description                      | Example  
------------------|---------------|---------------|-------------|------------  
KNNRecommender | ```userItemMatrix: Array<Array<string or number>> or null ```| void | This constructor takes a X x Y user item matrix as its argument. X[0] column represents the user id's and Y[0] the item labels. The cells in the matrix are expected to contain either -1 (dislike), 0 (no rating given) or 1 (like). The matrix can be null and you can use the addNewItemToDataset anda addNewUserToDataset methods for initializing the matrix | ```const kNNRecommender = new KNNRecommender([['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4','item 5', 'item 6', 'item 7'], ['user 1', 1, -1, 0, 0, -1, 1, 0], ['user 2', 1, -1, 0, 1, -1, 0, 0]]) ```
initializeRecommender | no arguments | ``` Promise<boolean> ``` | Initializes the recommender based on the provided user item matrix so we can start asking recommendations from it. If you add new items or users to the matrix and want the updates to affect the recommendations, you need to run this initialization again. This initialization is a heavy (roughly: O(n^3) + O(n * log(n)) operation. The method returns a Promise that resolves to true when the initialization is completed succesfully. | ```kNNRecommender.initializeRecommender().then(() => {... ```

# Performance

Performance tests were run on a 2014 Macbook Pro with a 2,2 GHz Quad-Core Intel Core i7 processor and 16 GM 1600 MHZ DDR3 memory.

Initialization times with different size matrices were as listed here:

Matrix size (users x items)      | Initialization time
------------------|---------------
100 x 100 | 40ms 
500 x 500 | 500ms 
1000 x 1000 | 23s
1000 x 50 | 1.2s
50 x 1000 | 80ms
50 x 10000 | 0.5s

Adding more users raises the initialization times radically. So if you find a way to divide your users into clusters, you can reduce the amount of user rows needed for providing recommendations for a certain user and thus provide recommendations faster.


# Contact

[Ohto Rainio](https://www.linkedin.com/in/ohtorainio/)

# Licence

MIT



