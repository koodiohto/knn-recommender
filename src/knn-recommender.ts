/** 
 * Copyright(c) 2021 Ohto Rainio
 * 
 * Licenced under MIT License(MIT)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files(the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
*/

type Similarity = {
    otherRowId: string;
    similarity: number;
};

type Recommendation = {
    itemId: string;
    recommenderUserId: string;
    similarityWithRecommender: number;
};

export default class KNNRecommender {

    private matrix: Array<Array<string | number>>

    private rowToRowSimilarityMap: {
        [key: string]: Array<Similarity>,
    } = {}

    private rowIdToRowNumberMap: {
        [key: string]: number,
    } = {}

    private columnIdToColumnNumberMap: {
        [key: string]: number,
    } = {}

    private initialized = false

    /**
     * Takes a user item (or item - item characterstic) matrix of size x*y where
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
     * This recommender can also work only based on non recommendations and recommendations (0's and 1's)
     * so it's not necessary to provide dislikes (-1).
     * You can initialize the recommender with a null-matrix and then fill it with addNewItemToDataset
     * and addNewUserToDataset methods before initializing it.
     * @param matrix = [['emptycorner', 'item 1', 'item 2', 'item 3'], ['user 1', 1, -1, 0], ['user 2', 0, -1, 1]]
     */
    constructor(matrix: Array<Array<string | number>> | null) {
        if (!matrix) {//allow initialization with and empty matrix to be filled later with addItems and addUsers methods.
            console.warn("Warning: Initializing knn-recommender with an empty user item matrix")
            this.matrix = new Array(new Array())
            this.matrix[0].push('emptycorner')
        } else {
            this.checkMatrix(matrix)
            this.matrix = matrix
        }
    }

    private checkMatrix(matrix: Array<Array<string | number>>) {
        if (!matrix || !matrix[0] || matrix[0].constructor !== Array ||
            (typeof matrix[0][1] !== "string")) {
            throw new TypeError(`Malformatted matrix. ` +
                `It should be a non zero two dimensional array in the format ` +
                `[['emptycorner', 'item 1', 'item 2', 'item 3'], ['user 1', 1, -1, 0], ['user 2', 0, -1, 1]]`)
        }
    }

    /**
     * Do the time consuming initializations. 
     * This is a heavy O(n^3) + O(n * log(n)) operation.
     */
    public initializeRecommender(): Promise<boolean> {
        return new Promise((resolve, reject) => {
            this.rowIdToRowNumberMap = {} //reinitialize
            this.columnIdToColumnNumberMap = {}
            this.calculateDistancesInZeroOneMatrixAndCreateRowIdToRowNumberAndColumnIdToColumnNumberMapsInChunks().then((value) => {
                this.initialized = true
                resolve(value)
            }).catch((error) => { reject(error) })
        });
    }

    /**
     * (Re)-initialize the recommender only for one userId.
     * This is significanly faster than initializing the recommender
     * for all users, so you should use this method when possible.
     * 
     * @param userId 
     */
    public initializeRecommenderForUserId(userId: string) {
        this.initializeRecommenderForRowId(userId)
    }

    /**
     * (Re)-initialize the recommender only for one itemid.
     * This is significanly faster than initializing the recommender
     * for all items, so you should use this method when possible.
     * @param itemId 
     */
    public initializeRecommenderForItemId(itemId: string) {
        this.initializeRecommenderForRowId(itemId)
    }

    private initializeRecommenderForRowId(rowId: string) {
        this.rowIdToRowNumberMap[rowId] = 0 //reinitialize this, we are counting on javascript intepretating 0 as false in our checks..A bit risky.
        for (let i = 0; i < this.matrix.length; i++) {
            if (this.matrix[i][0] === rowId) {
                this.calculateDistancesInZeroOneMatrixAndCreateRowIdToRowNumberAndColumnIdToColumnNumberMaps(i, i + 1)
                this.initialized = true
                return
            }
        }
        throw Error(`Id ${rowId} not found as a row in the matrix!`)
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
        Array<Similarity> {
        return this.getNNearestNeighboursForRowId(userId, amountOfDesiredNeighbours)
    }

    /**
     * Returns a sorted list of the n most similar items to the given itemId. 
     * The elements in the list contain objects in the form {otherItemId, similarity}.
     * E.g. [{otherItemId: 'Item 2', similarity: 0.53}, {otherItemId: 'Item 3', similarity: 0.4}]
     * 
     * @param itemId
     * @param amountOfDesiredNeighbours if not specified get similarity with all other users.
     * @returns 
     */
    public getNNearestNeighboursForItemId(itemId: string, amountOfDesiredNeighbours: number = -1):
        Array<Similarity> {
        return this.getNNearestNeighboursForRowId(itemId, amountOfDesiredNeighbours)
    }

    private getNNearestNeighboursForRowId(rowId: string, amountOfDesiredNeighbours: number = -1):
        Array<Similarity> {
        this.checkInitiated()
        let userSimilarities: Array<Similarity> = this.rowToRowSimilarityMap[rowId]

        if (!userSimilarities) {
            throw Error(`Similarities not initialized for rowId: ${rowId}`)
        }

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
       * @param amountOfDesiredNearestNeighboursToUse defaults to 3
       * @returns An array containing the recommendations or an empty array if no recommendations can be generated from the data
       * e.g. [{itemId: 'item 1', recommenderUserId: 'user 3', similarityWithRecommender: 0.6},
       * {itemId: 'item 1', recommenderUserId: 'user 2', similarityWithRecommender: 0.4}
       * {itemId: 'item 3', recommenderUserId: 'user 2', similarityWithRecommender: 0.4}, null]
       */
    public generateNNewRecommendationsForUserId(userId: string,
        amountOfDesiredNewRecommendations: number = 1,
        amountOfDesiredNearestNeighboursToUse: number = 3): Array<Recommendation> {
        this.checkInitiated()
        return this.generateNNewRecommendationsForUserIdInternal(userId, false,
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
      * @param amountOfDesiredNearestNeighboursToUse defaults to 3
      * @returns An array containing the recommendations or an empty array if no recommendations can be generated from the data
      * e.g. [{itemId: 'item 1', recommenderUserId: 'user 3', similarityWithRecommender: 0.6},
      * itemId: 'item 3', recommenderUserId: 'user 2', similarityWithRecommender: 0.4}, null]
      */
    public generateNNewUniqueRecommendationsForUserId(userId: string,
        amountOfDesiredNewRecommendations: number = 1,
        amountOfDesiredNearestNeighboursToUse: number = 3): Array<Recommendation> {
        this.checkInitiated()
        return this.generateNNewRecommendationsForUserIdInternal(userId, true,
            amountOfDesiredNewRecommendations,
            amountOfDesiredNearestNeighboursToUse)
    }

    /**
     * Update the liking value for a certain user and item.
     * @param userId 
     * @param itemId 
     */
    public addLikeForUserToAnItem(userId: string, itemId: string) {
        this.updateMatrixForRowIdAndColumnId(userId, itemId, 1)
    }

    /**
     * Update the disliking value for a certain user and item.
     * @param userId 
     * @param itemId 
     */
    public addDislikeForUserToAnItem(userId: string, itemId: string) {
        this.updateMatrixForRowIdAndColumnId(userId, itemId, -1)
    }

    /**
     * Add the charasteristic for a certain item and item charasteristic.
     * @param itemId 
     * @param itemCharasteristicId 
     */
    public addCharacteristicForItem(itemId: string, itemCharasteristicId: string) {
        this.updateMatrixForRowIdAndColumnId(itemId, itemCharasteristicId, 1)
    }

    /**
     * Remove the charasteristic for a certain item and item charasteristic.
     * @param itemId 
     * @param itemCharasteristicId 
     */
    public removeCharacteristicForItem(itemId: string, itemCharasteristicId: string) {
        this.updateMatrixForRowIdAndColumnId(itemId, itemCharasteristicId, 0)
    }

    /**
     * Update the liking value for a certain user and item.
     * NOTE: This method does not invoke an automatic recalculation of the 
     * user similarities. You need to tricker that manually if you wish by running
     * initializeRecommender-method
     * 
     * @param rowId 
     * @param columnId 
     * @param value -1, 0 or 1
     */
    public updateMatrixForRowIdAndColumnId(rowId: string, columnId: string, value: number) {
        if (!this.rowIdToRowNumberMap[rowId] || !this.columnIdToColumnNumberMap[columnId]) {
            throw new Error("userId or itemId not valid when updating user's value. " +
                "Have you initialized the recommender after adding new items or users?")
        }
        this.matrix[this.rowIdToRowNumberMap[rowId]][this.columnIdToColumnNumberMap[columnId]] = value
    }

    /**
     * Add a new user row to the data set
     * NOTE: This method does not invoke an automatic recalculation of the
     * user similarities. You need to tricker that manually if you wish by running
     * initializeRecommender-method
     * @param userRow ['user x', 1, 0, -1, ...]
     */
    public addNewUserToDataset(userRow: Array<string | number>) {
        if (!userRow || userRow.length != this.matrix[0].length) {
            throw new Error("The row to be added doesn't have the same amount of columns as the other items in the matrix")
        } else if (typeof userRow[0] != "string") {
            throw new Error("The row to be added isn't in the correct format that should be ['user id', 0, 1, ...]")
        } else if (this.rowIdToRowNumberMap[userRow[0]]) {
            throw new Error(`A row for the given id: ${userRow[0]} already exists in the user item matrix. Can't add a second row for the same user id. `)
        }
        this.rowIdToRowNumberMap[userRow[0]] = this.matrix.push(userRow) - 1
    }

    /**
     * Convenience method to add an empty user to data set with only user id.
     * All the recommendations are initialized with 0
     * @param userId 
     * */
    public addNewEmptyUserToDataset(userId: string) {
        this.addNewEmptyRowToDataset(userId)
    }

    /**
     * Convenience method to add an empty item to data set with only item id.
     * All the charasteristics are initialized with 0
     * @param itemId
     * */
    public addNewEmptyItemAsRowToDataset(itemId: string) {
        this.addNewEmptyRowToDataset(itemId)
    }

    private addNewEmptyRowToDataset(rowId: string) {
        let userArray = new Array<string | number>(this.matrix[0].length)
        userArray[0] = rowId
        for (let i = 1; i < this.matrix[0].length; i++) {
            userArray[i] = 0
        }
        this.addNewUserToDataset(userArray)
    }

    /**
     * Add a new item to the user item matrix and initalize all user recommendations
     * with value 0 for the new item.
     * NOTE: This method does not invoke an automatic recalculation of the
     * user similarities. You need to tricker that manually if you wish by running
     * initializeRecommender-method
     * @param itemId 
     */
    public addNewItemToDataset(itemId: string) {
        this.addNewColumnToDataset(itemId)
    }

    /**
     * Add a new item charasteristic to the item-item charasteristic matrix and initalize all 
     * items with value 0 for the new item charasteristic.
     * @param charasteristic 
     */
    public addNewItemCharacteristicToDataset(charasteristic: string) {
        this.addNewColumnToDataset(charasteristic)
    }

    private addNewColumnToDataset(columnId: string) {
        this.columnIdToColumnNumberMap[columnId] = this.matrix[0].push(columnId) - 1
        for (let i = 1; i < this.matrix.length; i++) {//initialize all recommendations with zeros for the new item
            this.matrix[i].push(0)
        }
    }

    /**
     * Get all the recommendations for certain user id. 
     * You can use this method together with getNNearestNeighboursForUserId 
     * to manually generate recommendations
     * for one user based on the recommedations of other users.
     * @param userId 
     * @returns e.g. ['user 1', 1, 0, -1, 0]
     */
    public getAllRecommendationsForUserId(userId: string): Array<string | number> {
        const rowNumber = this.rowIdToRowNumberMap[userId]
        if (!rowNumber) {
            throw new Error(`Invalid or non initialized user id ${userId}`)
        }
        return this.matrix[rowNumber]
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
     * @param amountOfDesiredNearestNeighboursToUse defaults to 3
     * @returns An array containing the recommendations or an empty array if no recommendations can be generated from the data
     * e.g. [{itemId: 'item 1', recommenderUserId: 'user 3', similarityWithRecommender: 0.6}, 
     * itemId: 'item 3', recommenderUserId: 'user 2', similarityWithRecommender: 0.4}, null]
     */
    private generateNNewRecommendationsForUserIdInternal(userId: string,
        onlyUnique: boolean,
        amountOfDesiredNewRecommendations: number = 1,
        amountOfDesiredNearestNeighboursToUse: number = 3
    ): Array<Recommendation> {
        const userRecommendations = this.getAllRecommendationsForUserId(userId)
        const userSimilarities = this.getNNearestNeighboursForUserId(userId, amountOfDesiredNearestNeighboursToUse)

        let newRecommendations = new Array<Recommendation>(amountOfDesiredNewRecommendations)
        let newRecommendationCounter = 0

        //embrace duplicate recommendations (several similar users have recommended something)
        let recommendationsAlreadyIncluded: {
            [key: number]: boolean,
        } = {}

        for (let i = 0; i < userSimilarities.length; i++) {
            const otherUsersRecommendations = this.getAllRecommendationsForUserId(userSimilarities[i].otherRowId)
            for (let j = 1; j < userRecommendations.length; j++) {
                if ((!onlyUnique || !recommendationsAlreadyIncluded[j]) &&
                    otherUsersRecommendations[j] === 1 && userRecommendations[j] === 0) {//the other user has liked this item and the current user has neither liked/disliked it.
                    newRecommendations[newRecommendationCounter] = {
                        itemId: <string>this.matrix[0][j],
                        recommenderUserId: userSimilarities[i].otherRowId,
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

    private checkInitiated(): void {
        if (!this.initialized) {
            throw new Error("Recommender not initialized!")
        }
    }

    /**
     * Run initialization in chunks.
     * 
     * @returns return a promise that resolves to true when initalization is done.
     */
    private calculateDistancesInZeroOneMatrixAndCreateRowIdToRowNumberAndColumnIdToColumnNumberMapsInChunks(): Promise<boolean> {
        return new Promise((resolve, reject) => {
            this.chunkIntermediator(1, this.matrix.length, resolve, reject)
        });
    }

    /**
     * Run calculateDistancesInZeroOneUserItemMatrixAndCreateUserToRowAndItemToColumnMap method
     * in chunks. Let the javascript event loop run through after each chunk to check for
     * other stuff in the event loop so we don't block the event loop. 
     * A new chunkIntermediator-method is added to the timer phase of the next event loop.
     * This is a "threaded" version of the initialization.
     * 
     * @param startIndex 
     * @param totalRows 
     * @param resolve 
     * @param reject 
     * @returns 
     */
    private chunkIntermediator(startIndex: number, totalRows: number, resolve: Function, reject: Function) {
        const CHUNK_SIZE = 3
        const rowsLenghtOrIPlusChunkSize = (startIndex + CHUNK_SIZE) > totalRows ? totalRows : (startIndex + CHUNK_SIZE)
        try {
            this.calculateDistancesInZeroOneMatrixAndCreateRowIdToRowNumberAndColumnIdToColumnNumberMaps(startIndex, rowsLenghtOrIPlusChunkSize)
        } catch (error) {
            reject(error)
            return;
        }

        if (rowsLenghtOrIPlusChunkSize < totalRows) {
            setTimeout(() => this.chunkIntermediator(startIndex + CHUNK_SIZE, totalRows, resolve, reject), 0)
        } else {
            this.initialized = true
            resolve(true)
        }
    }

    /**
     * This is a heavy (roughly: O(n^3) + O(n * log(n)) operation. 
     */
    private calculateDistancesInZeroOneMatrixAndCreateRowIdToRowNumberAndColumnIdToColumnNumberMaps(startAtRow: number, endAtRow: number): void {
        const rows = this.matrix.length
        const columns = this.matrix[0].length

        let itemIdToColumnNumberMapInitiated = false

        for (let i = startAtRow; i < endAtRow; i++) {//first row is item names
            let rowToOtherRowsSimilarityList: Array<Similarity> = Array(rows - 2)
            let rowToOtherRowsCounter = 0
            //Go through all the rows to match the row with all the rest of the rows
            for (let i2 = 1; i2 < rows; i2++) {
                if (i === i2) {//don't compare the row to itself
                    continue
                }
                let similarRatings = 0
                let ratingsDoneByEitherRow = 0
                for (let j = 1; j < columns; j++) {//first column contains the row id's, start with second
                    if (!itemIdToColumnNumberMapInitiated) {
                        this.columnIdToColumnNumberMap[<string>this.matrix[0][j]] = j
                    }
                    if (this.matrix[i][j] !== -1 && this.matrix[i][j] !== 0
                        && this.matrix[i][j] !== 1) {
                        throw new RangeError(`Element in matrix was invalid, either not a` +
                            ` number at all or not a -1, 0, or 1. The invalid value  ` +
                            `at index [${i}][${j}] is: ${this.matrix[i][j]}`)
                    } else if (this.matrix[i][j] !== 0 && this.matrix[i][j] === this.matrix[i2][j]) {
                        similarRatings++
                        ratingsDoneByEitherRow++
                    } else if (this.matrix[i][j] !== 0 || this.matrix[i2][j] !== 0) {
                        ratingsDoneByEitherRow++
                    }
                }
                itemIdToColumnNumberMapInitiated = true // initiate this only once on the first run
                let jaccardSimilarity = ratingsDoneByEitherRow !== 0 ? similarRatings / ratingsDoneByEitherRow : 0

                if (typeof this.matrix[i2][0] !== "string") {
                    throw new TypeError(`Malformatted matrix. Element at` +
                        `at index [${i2}][${0}] is not a string (describing a user/item id). ` +
                        `The invalid element is: ${this.matrix[i2][0]}`)
                }

                rowToOtherRowsSimilarityList[rowToOtherRowsCounter] = { otherRowId: <string>this.matrix[i2][0], similarity: jaccardSimilarity }
                rowToOtherRowsCounter++
            }
            if (typeof this.matrix[i][0] !== "string") {
                throw new TypeError(`Malformatted matrix. Element at` +
                    `at index [${i}][${0}] is not a string (describing a user/item id). ` +
                    `The invalid element is: ${this.matrix[i][0]}`)
            } else if (this.rowIdToRowNumberMap[this.matrix[i][0]]) {
                throw new Error(`Malformatted matrix. The matrix contains two rows with the same id: ${this.matrix[i][0]}`)
            }

            this.rowIdToRowNumberMap[this.matrix[i][0]] = i
            this.rowToRowSimilarityMap[this.matrix[i][0]] = this.sortRowToOtherRowsSimilarityListByRowToRowSimilarityDescending(rowToOtherRowsSimilarityList)
        }
    }

    private sortRowToOtherRowsSimilarityListByRowToRowSimilarityDescending(userToOtherUsersSimilarityList: Array<Similarity>): Array<Similarity> {
        return userToOtherUsersSimilarityList.sort((a, b) => (a.similarity > b.similarity) ? -1 : 1)
    }
}