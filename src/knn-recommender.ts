type UserSimilarity = {
    otherUserId: string;
    similarity: number;
};

export default class KNNRecommender {


    private userItemMatrix: string | number[][]

    //TODO:
    //private userItemMatrixNewAdditions: string | number[][]


    private userToUserSimilarityMap: {
        [key: string]: Array<UserSimilarity>,
    } = {}

    private userToRowNumberMap: {
        [key: string]: number,
    } = {}


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
    }

    /**
     * Returns a sorted list of the x most similar users to the given userId. 
     * The elements in the list contain objects in the form {otherUserId, similarity}.
     * E.g. [{otherUserId: 'User 2', similarity: 0.53}, {otherUserId: 'User 3', similarity: 0.4}]
     * 
     * @param userId 
     * @param amountOfDesiredNeighbours if not specified get similarity with all other users.
     * @returns 
     */
    public getXNearestNeighboursForUserId(userId: string, amountOfDesiredNeighbours: number = -1):
        Array<UserSimilarity> {
        let userSimilarities: Array<UserSimilarity> = this.userToUserSimilarityMap[userId]

        if (amountOfDesiredNeighbours !== -1 && userSimilarities.length > amountOfDesiredNeighbours) {
            return userSimilarities.slice(0, amountOfDesiredNeighbours);
        }
        return userSimilarities
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