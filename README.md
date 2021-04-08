# knn-recommender
A pure JavaScript implementation of a K-nearest neighbour based collaborative filtering recommender primarily for like/dislike User-Item matrices. You can use the recommender e.g. for Item-Item characteristics matrices as well. So this library enables you to provide "You liked this, you might also like this" or "items similar to this item" recommendations. 

This library should run both in node and browser environments. This is an experimental implementation and is intended for fairly small size matrices (~1000 users). If you are looking for a more high performing (and properly threaded) library, I'd recommend you to check out [recommendationRacoon](https://github.com/guymorita/recommendationRaccoon).

The recommender takes a user item matrix of size X x Y where X[0] column represents the user id's and Y[0] the item labels. The cells in the matrix are expected to contain either -1 (dislike), 0 (no rating given) or 1 (like). This information can be used to calculate the similarity of users in the matrix based on jaccard similarity.

Example of a possible User-Item matrix:
```
[
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, -1, 0, 0, -1, 1, 0],
    ['user 2', 1, -1, 0, 1, -1, 0, 0]
]
```

Example of a possible Item-Item characteristics matrix:
```
[
    ['emptycorner', 'characteristic 1', 'characteristic 2', 'characteristic 3'],
    ['item 1', 1, 0, 1],
    ['item 2', 1, 1, 0]
]
```

Jaccard similarity calculates the common ratings between two users and divides that by the total ratings given by the users. The non-ratings are not considered when calculating the similarity of two users.

So if user X has a rating matrix (1, -1, 0, 0, -1, 1, 0) and Y (1, -1, 0, 1, -1, 0, 0) their Jaccard similarity is (1+1+1) / 5 = 3/5. 5 is effectively the number of the elements that at least one of the two users has either liked or disliked. Based on this we could provide a recommendation for user Y for item number 6 as Y hasn't expressed any preference for that and X has liked this item.

This recommender can also work only based on non recommendations and recommendations (0's and 1's so it's not necessary to provide dislikes (-1).

# Installation

```bash
npm install --save knn-recommender
```
You can also [download the javascript source for knn-recommender.js](https://github.com/koodiohto/knn-recommender/blob/main/dist/knn-recommender.js) directly from this repository. Or you can  [download the minified version compiled for npm-distribution](https://github.com/koodiohto/knn-recommender/blob/main/distfornpmpublishing/knn-recommender.js) that should run in browsers and node-environment.

# Basic usage

If you have the User-Item matrix available, you can initialize the recommender for all users.
```js
import KNNRecommender from 'knn-recommender';

const kNNRecommender = new KNNRecommender([['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4','item 5', 'item 6', 'item 7'], ['user 1', 1, -1, 0, 0, -1, 1, 0], ['user 2', 1, -1, 0, 1, -1, 0, 0]])
kNNRecommender.initializeRecommender().then(() => {
    const userRecommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 2')
    console.log(`new recommendation for user 2 ${userRecommendations[0].itemId}`)
})
```

Or you can start filling the items and users to the matrix one by one and also initialize the recommender only for only certain users. Initializing the recommender only for one user is significantly faster, so you should do that if you only provide recommendations for this particular user.

```js
import KNNRecommender from 'knn-recommender';

const kNNRecommender = new KNNRecommender(null)
kNNRecommender.addNewItemToDataset('item 1')
kNNRecommender.addNewItemToDataset('item 2')
kNNRecommender.addNewEmptyUserToDataset('user 1')
kNNRecommender.addNewEmptyUserToDataset('user 2')

kNNRecommender.addDislikeForUserToAnItem('user 1', 'item 1')
kNNRecommender.addLikeForUserToAnItem('user 1', 'item 2')
kNNRecommender.addDislikeForUserToAnItem('user 2', 'item 1')

kNNRecommender.initializeRecommenderForUserId('user 2')

const user2Recommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 2', 1)

//should print 'item 2'
console.log(`new recommendation for user 2 ${user2Recommendations[0].itemId}`)

kNNRecommender.addNewItemToDataset('item 3')
kNNRecommender.addLikeForUserToAnItem('user 2', 'item 3')

kNNRecommender.initializeRecommenderForUserId('user 1')

const user1Recommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 1', 1)

//should print 'item 3'
console.log(`new recommendation for user 1 ${user1Recommendations[0].itemId}`)

```

Sidenote:
If you use node without babel, you have to import the module like this:
```js
const recommender = KNNRecommender('knn-recommender');
const kNNRecommender = new KNNRecommender.default(null)
kNNRecommender.addNewItemToDataset('item 1')
...
```

# API

Scroll to the right to see all the columns...

Method             | Arguments          | Returns      | Description                      | Example  
------------------|---------------|---------------|-------------|------------  
KNNRecommender | ```userItemMatrix: Array<Array<string or number>> or null ```| void | This constructor takes a X x Y user item matrix as its argument. X[0] column represents the user id's and Y[0] the item labels. The cells in the matrix are expected to contain either -1 (dislike), 0 (no rating given) or 1 (like). The matrix can be null and you can use the addNewItemToDataset anda addNewUserToDataset methods for initializing the matrix | ```const kNNRecommender = new KNNRecommender([['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4','item 5', 'item 6', 'item 7'], ['user 1', 1, -1, 0, 0, -1, 1, 0], ['user 2', 1, -1, 0, 1, -1, 0, 0]]) ```
initializeRecommender | no arguments | ``` Promise<boolean> ``` | Initializes the recommender for all users based on the provided user item matrix so we can start asking recommendations from it. If you add new items or users to the matrix and want the updates to affect the recommendations, you need to run this initialization again. This initialization is a heavy (roughly: O(n^3) + O(n * log(n)) operation. The method returns a Promise that resolves to true when the initialization is completed successfully. | ```kNNRecommender.initializeRecommender().then(() => {... ```
initializeRecommenderForUserId | ```userId: string ``` | void | (Re)-initialize the recommender only for one userId. This is significantly faster than initializing the recommender for all users, so you should use this method when possible. | ```kNNRecommender.initializeRecommenderForUserId('user 1')```
generateNNewRecommendationsForUserId | ```userId: string, amountOfDesiredNewRecommendations: number = 1, amountOfDesiredNearestNeighboursToUse: number = 3 ``` | ``` Array<Recommendation> ``` | Try to generate the desired amount of new recommendations for a user based on what similar users have liked. The method starts with the most similar user and collects all the likings from him/her where the current user hasn't expressed their preference yet. If the amount of desired recommendations hasn't been fulfilled yet, it proceeds to the second most similar user and so on. The method might add the same recommendation twice if an item has been recommended by several similar users. If you want to have these potential multi recommendations excluded use the method generateNNewUniqueRecommendationsForUserId instead. Returns an array containing the recommendations or an empty array if no recommendations can be generated from the data e.g. ```[{itemId: 'item 1', recommenderUserId: 'user 3', similarityWithRecommender: 0.6}, {itemId: 'item 1', recommenderUserId: 'user 2', similarityWithRecommender: 0.4} {itemId: 'item 3', recommenderUserId: 'user 2', similarityWithRecommender: 0.4}, null] ``` | ```const userRecommendations = kNNRecommender.generateNNewRecommendationsForUserId('user 3', 2, 3); console.log(`${userRecommendations[0].itemId} ${userRecommendations[0].recommenderUserId} {userRecommendations[0].similarityWithRecommender}`)```
generateNNewUniqueRecommendationsForUserId | ```userId: string, amountOfDesiredNewRecommendations: number = 1, amountOfDesiredNearestNeighboursToUse: number = 3 ``` | ``` Array<Recommendation> ``` | Try to generate the desired amount of new recommendations for a user based on what similar users have liked. The method starts with the most similar user and collects all the likings from him/her where the current user hasn't expressed their preference yet. If the amount of desired recommendations hasn't been fulfilled yet, it proceeds to the second most similar user and so on. The method doesn't add the same recommendation twice even if it would be recommended by several users. If you want to have these potential multi recommendations included use the method generateNNewRecommendationsForUserId instead. Returns an array containing the recommendations or an empty array if no recommendations can be generated from the data e.g. ```[{itemId: 'item 1', recommenderUserId: 'user 3', similarityWithRecommender: 0.6}, itemId: 'item 3', recommenderUserId: 'user 2', similarityWithRecommender: 0.4}, null] ``` | ```const userRecommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 3', 2, 3); console.log(`${userRecommendations[0].itemId} ${userRecommendations[0].recommenderUserId} {userRecommendations[0].similarityWithRecommender}`)```
getNNearestNeighboursForUserId | ```userId: string, amountOfDesiredNeighbours: number = -1 ``` | ``` Array<UserSimilarity> ``` | Returns a sorted list of the n most similar users to the given userId. The elements in the list contain objects in the form {otherUserId, similarity}. E.g. ```[{otherUserId: 'User 2', similarity: 0.53}, {otherUserId: 'User 3', similarity: 0.4}] ```| ```  const user1ToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 1')```
getAllRecommendationsForUserId | ```userId: string ``` | ``` Array<string or number> ``` | Get all the recommendations for certain user id. You can use this method together with getNNearestNeighboursForUserId to manually generate recommendations for one user based on the recommendations of other users. Returns e.g. ```['user 1', 1, 0, -1, 0]``` | ```  const allUserRecommendations = kNNRecommender.getAllRecommendationsForUserId('user 1')```
addLikeForUserToAnItem | ```userId: string, itemId: string ``` | void | Update the liking value for a certain user and item. NOTE: This method does not invoke an automatic recalculation of the user similarities. You need to tricker that manually if you wish by running initializeRecommender-method | ```kNNRecommender.addLikeForUserToAnItem('user 1', 'item 2')```
addDislikeForUserToAnItem | ```userId: string, itemId: string ``` | void | Update the disliking value for a certain user and item. NOTE: This method does not invoke an automatic recalculation of the user similarities. You need to tricker that manually if you wish by running initializeRecommender-method | ```kNNRecommender.addDislikeForUserToAnItem('user 1', 'item 2')```
addNewUserToDataset | ```userRow: Array<string or number>``` | void | Add a new user row to the data set. NOTE: This method does not invoke an automatic recalculation of the user similarities. You need to tricker that manually if you wish by running initializeRecommender-method. | ```kNNRecommender.addNewUserToDataset(['user x', 1, 0, -1, ...])```
addNewEmptyUserToDataset | ```userId: string``` | void | Convenience method to add an empty user to data set with only user id. All the recommendations are initialized with 0. NOTE: This method does not invoke an automatic recalculation of the user similarities. You need to tricker that manually if you wish by running initializeRecommender-method. | ```kNNRecommender.addNewEmptyUserToDataset('user 3')```
addNewItemToDataset | ```itemId: string``` | void | Add a new item to the user item matrix and initialize all user recommendations with value 0 for the new item. NOTE: This method does not invoke an automatic recalculation of the user similarities. You need to tricker that manually if you wish by running initializeRecommender-method | ```kNNRecommender.addNewItemToDataset('item 8')```

# Performance

Performance tests were run on a 2014 Macbook Pro with a 2,2 GHz Quad-Core Intel Core i7 processor and 16 GM 1600 MHZ DDR3 memory.

Initialization times with different size matrices were as listed here:

Matrix size (users x items)      | Initialization time
------------------|---------------
100 x 100 | 40ms 
500 x 500 | 2.8s 
1000 x 1000 | 23s
1000 x 50 | 1.2s
50 x 1000 | 80ms
50 x 10000 | 0.5s

Adding more users raises the initialization times radically. So if you find a way to divide your users into clusters, you can reduce the amount of user rows needed for providing recommendations for a certain user and thus provide recommendations faster. Or you can initialize the recommender only for individual users.

# Contact

[Ohto Rainio](https://www.linkedin.com/in/ohtorainio/)

# Licence

MIT
