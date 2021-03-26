type UserSimilarity = {
    otherUserId: string;
    similarity: number;
};

type Recommendation = {
    itemId: string;
    recommenderUserId: string;
    similarityWithRecommender: number;
};



export default class KNNRecommender {


    private userItemMatrix: string | number[][]

    private userToUserSimilarityMap: {
        [key: string]: Array<UserSimilarity>,
    } = {}

    private userToRowNumberMap: {
        [key: string]: number,
    } = {}

    //TODO: initialize this also
    private itemNameToColumnNumberMap: {
        [key: string]: number,
    } = {}

    private initialized = false

    /**
     * Takes a user item matrix of size x*y where
     * x[0] column represents the user id's and 
     * y[0] the item labels. The cells in the matrix 
     * are expected to contain either -1 (dislike), 
     * 0 (no rating given) or 1 (like). This information
     * can be used to calculate the similarity of users
     * in the matrix based on jaccard similarity.
     * The Jaccard similarity calculates the common 
     * ratings between two users and divides that
     * by the total ratings given by the users.
     * The non-ratings are not considered when 
     * calculating the similarity of two users.
     * So if user X has a rating matrix (1, -1, 0, 0, -1, 1, 0)
     * and Y (1, -1, 0, 1, -1, 0, 0) their Jaccard similarity
     * is (1+1+1) / 5 = 3/5. 5 is effectively the number of the elements 
     * that at least one of the two users has either liked or disliked.
     * @param userItemMatrix = [['emptycorner', 'item 1', 'item 2', 'item 3'], ['user 1', 1, -1, 0], ['user 2', 0, -1, 1]]
     */
    constructor(userItemMatrix: string | number[][]) {
        this.checkUserItemMatrix(userItemMatrix)
        this.userItemMatrix = userItemMatrix
    }

    private checkUserItemMatrix(userItemMatrix: string | number[][]) {
        if (!userItemMatrix || userItemMatrix.length < 2 || !userItemMatrix[0] || userItemMatrix[0].length < 2 ||
            (typeof userItemMatrix[0][1] !== "string") || (typeof userItemMatrix[1][0] !== "string") ||
            (typeof userItemMatrix[1][1] !== "number")) {
            throw new TypeError(`Malformatted user item matrix. ` +
                `It should be a non zero two dimensional array in the format ` +
                `[['emptycorner', 'item 1', 'item 2', 'item 3'], ['user 1', 1, -1, 0], ['user 2', 0, -1, 1]]`)
        }
    }

    /**
     * Do the time consuming initializations. 
     * This is a heavy O(n^3) + O(n) operation, so it's recommended
     * to run it in a thread provided by your running environment.
     * This library tries to be agnostic to the Javascript engine used and 
     * thus this is not threaded here.
     */
    public initializeKNNRecommenderForZeroOneUserMatrix(): void {
        this.calculateDistancesInZeroOneUserItemMatrixAndCreateUserToRowMap()
        this.initialized = true
    }

    /**
     * Returns a sorted list of the n most similar users to the given userId. 
     * The elements in the list contain objects in the form {otherUserId, similarity}.
     * E.g. [{otherUserId: 'User 2', similarity: 0.53}, {otherUserId: 'User 3', similarity: 0.4}]
     * 
     * @param userId 
     * @param amountOfDesiredNeighbours if not specified get similarity with all other users.
     * @returns 
     */
    public getNNearestNeighboursForUserId(userId: string, amountOfDesiredNeighbours: number = -1):
        Array<UserSimilarity> {
        this.checkInitiated()
        let userSimilarities: Array<UserSimilarity> = this.userToUserSimilarityMap[userId]

        if (amountOfDesiredNeighbours !== -1 && userSimilarities.length > amountOfDesiredNeighbours) {
            return userSimilarities.slice(0, amountOfDesiredNeighbours);
        }
        return userSimilarities
    }

    /**
   * Try to generate the desired amount of new recommendations for a user
   * based on what similar users have liked.
   * The method starts with the most similar user and collects all the
   * likings from him/her where the current user hasn't expressed their
   * preference yet. If the amount of desired recommendations hasn't been
   * fulfilled yet, it proceededs to the second most similar user and so on.
   * The method might add the same recommendation twice if an item has been
   * recommended by several similar users. If you want to have these potential multi recommendations
   * exluded use the method generateNNewUniqueRecommendationsForUserId instead.
   * @param userId
   * @param amountOfDesiredNewRecommendations defaults to 1
   * @param amountOfDesiredNearestNeighboursToUse defaults to 1
   * @returns An array containing the recommendations or null's if no recommendations can be generated from the data
   * e.g. [{itemId: 'item 1', recommenderUserId: 'user 3', similarityWithRecommender: 0.6},
   * {itemId: 'item 1', recommenderUserId: 'user 2', similarityWithRecommender: 0.4}
   * {itemId: 'item 3', recommenderUserId: 'user 2', similarityWithRecommender: 0.4}, null]
   */
    public generateNNewRecommendationsForUserId(userId: string,
        amountOfDesiredNewRecommendations: number = 1,
        amountOfDesiredNearestNeighboursToUse: number = 1): Array<Recommendation> {
        this.checkInitiated()
        return this.generateXNewRecommendationsForUserIdInternal(userId, false,
            amountOfDesiredNewRecommendations,
            amountOfDesiredNearestNeighboursToUse)
    }

    /**
      * Try to generate the desired amount of new unique recommendations for a user
      * based on what similar users have liked.
      * The method starts with the most similar user and collects all the
      * likings from him/her where the current user hasn't expressed their
      * preference yet. If the amount of desired recommendations hasn't been
      * fulfilled yet, it proceededs to the second most similar user and so on.
      * The method doesn't add same recommendation twice even if it would be
      * recommended by several users. If you want to have these potential multi recommendations
      * included use the method generateNNewRecommendationsForUserId instead.
      * @param userId
      * @param amountOfDesiredNewRecommendations defaults to 1
      * @param amountOfDesiredNearestNeighboursToUse defaults to 1
      * @returns An array containing the recommendations or null's if no recommendations can be generated from the data
      * e.g. [{itemId: 'item 1', recommenderUserId: 'user 3', similarityWithRecommender: 0.6},
      * itemId: 'item 3', recommenderUserId: 'user 2', similarityWithRecommender: 0.4}, null]
      */
    public generateNNewUniqueRecommendationsForUserId(userId: string,
        amountOfDesiredNewRecommendations: number = 1,
        amountOfDesiredNearestNeighboursToUse: number = 1): Array<Recommendation> {
        this.checkInitiated()
        return this.generateXNewRecommendationsForUserIdInternal(userId, true,
            amountOfDesiredNewRecommendations,
            amountOfDesiredNearestNeighboursToUse)
    }



    /**
     * Try to generate the desired amount of new unique recommendations for a user
     * based on what similar users have liked.
     * The method starts with the most similar user and collects all the 
     * likings from him/her where the current user hasn't expressed their
     * preference yet. If the amount of desired recommendations hasn't been
     * fulfilled yet, it proceededs to the second most similar user and so on.
     * The method doesn't add same recommendation twice even if it would be 
     * recommended by several users. If you want to have these potential multi recommendations
     * included use the method generateNNewRecommendationsForUserId instead.
     * @param userId 
     * @param onlyUnique
     * @param amountOfDesiredNewRecommendations defaults to 1
     * @param amountOfDesiredNearestNeighboursToUse defaults to 1
     * @returns An array containing the recommendations or null's if no recommendations can be generated from the data
     * e.g. [{itemId: 'item 1', recommenderUserId: 'user 3', similarityWithRecommender: 0.6}, 
     * itemId: 'item 3', recommenderUserId: 'user 2', similarityWithRecommender: 0.4}, null]
     */
    private generateXNewRecommendationsForUserIdInternal(userId: string,
        onlyUnique: boolean,
        amountOfDesiredNewRecommendations: number = 1,
        amountOfDesiredNearestNeighboursToUse: number = 1
    ): Array<Recommendation> {
        const userRecommendations = this.getAllRecommendationsForUserId(userId)
        const userSimilarities = this.getNNearestNeighboursForUserId(userId, amountOfDesiredNearestNeighboursToUse)

        let newRecommendations = new Array<Recommendation>(amountOfDesiredNewRecommendations)
        let newRecommendationCounter = 0

        //embrace duplicate recommendations, if many similar users have recommended something
        let recommendationsAlreadyIncluded: {
            [key: number]: boolean,
        } = {}

        for (let i = 0; i < userSimilarities.length; i++) {
            const otherUsersRecommendations = this.getAllRecommendationsForUserId(userSimilarities[i].otherUserId)
            for (let j = 1; j < userRecommendations.length; j++) {
                if ((!onlyUnique || !recommendationsAlreadyIncluded[j]) &&
                    otherUsersRecommendations[j] === 1 && userRecommendations[j] === 0) {//the other user has liked this item and the current user has neither liked/disliked it.
                    newRecommendations[newRecommendationCounter] = {
                        itemId: <string>this.userItemMatrix[0][j],
                        recommenderUserId: userSimilarities[i].otherUserId,
                        similarityWithRecommender: userSimilarities[i].similarity
                    }
                    newRecommendationCounter++
                    if (newRecommendationCounter >= amountOfDesiredNewRecommendations) {
                        //We found enough recommendations, stop searching for more.
                        return newRecommendations
                    }
                    recommendationsAlreadyIncluded[j] = true
                }
            }
        }

        return newRecommendations
    }

    /**
     * Get all the recommendations for certain user id. 
     * You can use this method together with getNNearestNeighboursForUserId 
     * to manually generate recommendations
     * for one user based on the recommedations of other users.
     * @param userId 
     * @returns e.g. ['user 1', 1, 0, -1, 0]
     */
    public getAllRecommendationsForUserId(userId: string): string | number[] {
        const rowNumber = this.userToRowNumberMap[userId]
        if (!rowNumber) {
            throw new Error('Invalid user id')
        }
        return this.userItemMatrix[rowNumber]
    }

    private checkInitiated(): void {
        if (!this.initialized) {
            throw new Error("Recommender not initialized!")
        }
    }
    /**
     * This is a heavy (roughly: O(n^3) + O(n * log(n)) operation. 
     */
    private calculateDistancesInZeroOneUserItemMatrixAndCreateUserToRowMap(): void {
        const rows = this.userItemMatrix.length
        const columns = this.userItemMatrix[0].length

        for (let i = 1; i < rows; i++) {//first row is item names, start with second row
            let userToOtherUsersSimilarityList: Array<UserSimilarity> = Array(rows - 2)
            let userToOtherUsersCounter = 0
            //Go through all the rows to match the user with all the rest of the users
            for (let i2 = 1; i2 < rows; i2++) {
                if (i === i2) {//don't compare the user to themself
                    continue
                }
                let similarRatings = 0
                let ratingsDoneByEitherUser = 0
                for (let j = 1; j < columns; j++) {//first column contains the user name, start with second
                    if (this.userItemMatrix[i][j] !== -1 && this.userItemMatrix[i][j] !== 0
                        && this.userItemMatrix[i][j] !== 1) {
                        throw new RangeError(`Element in user item matrix was invalid, either not a` +
                            ` number at all or not a -1, 0, or 1. The invalid value  ` +
                            `at index [${i}][${j}] is: ${this.userItemMatrix[i][j]}`)
                    } else if (this.userItemMatrix[i][j] !== 0 && this.userItemMatrix[i][j] === this.userItemMatrix[i2][j]) {
                        similarRatings++
                        ratingsDoneByEitherUser++
                    } else if (this.userItemMatrix[i][j] !== 0 || this.userItemMatrix[i2][j] !== 0) {
                        ratingsDoneByEitherUser++
                    }
                }
                let jaccardSimilarity = similarRatings / ratingsDoneByEitherUser

                if (typeof this.userItemMatrix[i2][0] !== "string") {
                    throw new TypeError(`Malformatted user item matrix. Element at` +
                        `at index [${i2}][${0}] is not a string (describing a userid). ` +
                        `The invalid element is: ${this.userItemMatrix[i2][0]}`)
                }

                userToOtherUsersSimilarityList[userToOtherUsersCounter] = { otherUserId: <string>this.userItemMatrix[i2][0], similarity: jaccardSimilarity }
                userToOtherUsersCounter++
            }
            if (typeof this.userItemMatrix[i][0] !== "string") {
                throw new TypeError(`Malformatted user item matrix. Element at` +
                    `at index [${i}][${0}] is not a string (describing a userid). ` +
                    `The invalid element is: ${this.userItemMatrix[i][0]}`)
            }
            this.userToRowNumberMap[this.userItemMatrix[i][0]] = i
            this.userToUserSimilarityMap[this.userItemMatrix[i][0]] = this.sortUserToOtherUsersSimilarityListByUserToUserSimilarityDescending(userToOtherUsersSimilarityList)

        }
    }

    private sortUserToOtherUsersSimilarityListByUserToUserSimilarityDescending(userToOtherUsersSimilarityList: Array<UserSimilarity>): Array<UserSimilarity> {
        return userToOtherUsersSimilarityList.sort((a, b) => (a.similarity > b.similarity) ? -1 : 1)
    }
}