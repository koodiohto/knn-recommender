import KNNRecommender from '../src/knn-recommender';
import { expect } from 'chai';

const simpleUserItemMatrix: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, -1, 0, 0, -1, 1, 0],
    ['user 2', 1, -1, 0, 1, -1, 0, 0]
]


const threeUserItemMatrix: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, -1, 0, 0, -1, 1, 0],
    ['user 2', 1, -1, 0, 1, -1, 0, 0],
    ['user 3', 1, 0, 0, 1, 0, 1, 0]
]

const threeUserItemMatrixForFindingRecommendations: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, -1, 0, 0, -1, 1, 0],
    ['user 2', 1, -1, 0, 1, -1, 0, 0],
    ['user 3', 0, -1, 0, 1, -1, 0, 0]
]

const emptyUserItemMatrix: any[][] = []

const malformattedDataInUserItemMatrix: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, 'fdf', 0, 0, -1, 1, 0],
    ['user 2', 1, -1, 0, 1, -1, 0, 0],
    ['user 3', 1, 0, 0, 1, 0, 1, 0]
]

const malformattedUserIdANumber: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    [1, 1, 1, 0, 0, -1, 1, 0],
    ['user 2', 1, -1, 0, 1, -1, 0, 0],
    ['user 3', 1, 0, 0, 1, 0, 1, 0]
]

const malformattedSecondUserIdANumber: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, 1, 0, 0, -1, 1, 0],
    [1, 1, -1, 0, 1, -1, 0, 0],
    ['user 3', 1, 0, 0, 1, 0, 1, 0]
]

const malformattedItemName: any[][] = [
    ['emptycorner', 1, 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, 1, 0, 0, -1, 1, 0],
    ['user 2', 1, -1, 0, 1, -1, 0, 0],
    ['user 3', 1, 0, 0, 1, 0, 1, 0]
]


const wrongSizeUserItemMatrix: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, 0, 0, -1, 1, 0],
    ['user 2', 1, -1, 0, 1, -1, 0, 0],
    ['user 3', 1, 0, 0, 1, 0, 1, 0]
]

const wrongSize2UserItemMatrix: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, 1, 0, 0, -1, 1, 0],
    ['user 2', 1, 0, 1, -1, 0, 0],
    ['user 3', 1, 0, 0, 1, 0, 1, 0]
]


describe('basic test', () => {
    it('should get the first similar user correctly', () => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()
        const userToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 1', 1)

        expect(userToOtherUsersArray[0].otherUserId).to.equal('user 2');
        expect(userToOtherUsersArray[0].similarity).to.equal(3 / 5);
    })

    it('should get the similarities for three users correctly', () => {
        const kNNRecommender = new KNNRecommender(threeUserItemMatrix)
        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()
        const user1ToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 1', 2)

        expect(user1ToOtherUsersArray[0].otherUserId).to.equal('user 2');
        expect(user1ToOtherUsersArray[0].similarity).to.equal(3 / 5);
        expect(user1ToOtherUsersArray[1].otherUserId).to.equal('user 3');
        expect(user1ToOtherUsersArray[1].similarity).to.equal(2 / 5);

        const user2ToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 2')

        expect(user2ToOtherUsersArray[0].otherUserId).to.equal('user 1');
        expect(user2ToOtherUsersArray[0].similarity).to.equal(3 / 5);
        expect(user2ToOtherUsersArray[1].otherUserId).to.equal('user 3');
        expect(user2ToOtherUsersArray[1].similarity).to.equal(2 / 5);

        const user3ToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 3')

        expect(user3ToOtherUsersArray[0].otherUserId).to.equal('user 1');
        expect(user3ToOtherUsersArray[0].similarity).to.equal(2 / 5);
        expect(user3ToOtherUsersArray[1].otherUserId).to.equal('user 2');
        expect(user3ToOtherUsersArray[1].similarity).to.equal(2 / 5);
    })

    it('should get all the recommendations for user correctly', () => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()
        const userRecommendations = kNNRecommender.getAllRecommendationsForUserId('user 2')

        expect(userRecommendations[0]).to.equal('user 2');
        expect(userRecommendations[1]).to.equal(1);
        expect(userRecommendations[2]).to.equal(-1);
        expect(userRecommendations[3]).to.equal(0);

    })

    it('should get 3 new unique recommendations correctly for user', () => {
        const kNNRecommender = new KNNRecommender(threeUserItemMatrixForFindingRecommendations)
        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()
        const userRecommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 3', 3, 2)

        expect(userRecommendations[0].itemId).to.equal('item 1');
        expect(userRecommendations[0].recommenderUserId).to.equal('user 2');
        expect(userRecommendations[0].similarityWithRecommender).to.equal(3 / 4);
        expect(userRecommendations[1].itemId).to.equal('item 6');
        expect(userRecommendations[1].recommenderUserId).to.equal('user 1');
        expect(userRecommendations[1].similarityWithRecommender).to.equal(2 / 5);
        expect(userRecommendations[2]).to.equal(undefined);
    })

    it('should get 3 new (not unique) recommendations correctly for user', () => {
        const kNNRecommender = new KNNRecommender(threeUserItemMatrixForFindingRecommendations)
        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()
        const userRecommendations = kNNRecommender.generateNNewRecommendationsForUserId('user 3', 3, 2)

        expect(userRecommendations[0].itemId).to.equal('item 1');
        expect(userRecommendations[0].recommenderUserId).to.equal('user 2');
        expect(userRecommendations[0].similarityWithRecommender).to.equal(3 / 4);
        expect(userRecommendations[1].itemId).to.equal('item 1');
        expect(userRecommendations[1].recommenderUserId).to.equal('user 1');
        expect(userRecommendations[1].similarityWithRecommender).to.equal(2 / 5);
        expect(userRecommendations[2].itemId).to.equal('item 6');
        expect(userRecommendations[2].recommenderUserId).to.equal('user 1');
        expect(userRecommendations[2].similarityWithRecommender).to.equal(2 / 5);
    })

    it('should update item for user correctly', () => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()

        const userRecommendationsBefore = kNNRecommender.getAllRecommendationsForUserId('user 1')
        expect(userRecommendationsBefore[0]).to.equal('user 1');
        expect(userRecommendationsBefore[2]).to.equal(-1);

        kNNRecommender.updateUserItemMatrixForUserId('user 1', 'item 2', 1)

        const userRecommendationsAfter = kNNRecommender.getAllRecommendationsForUserId('user 1')
        expect(userRecommendationsAfter[0]).to.equal('user 1');
        expect(userRecommendationsAfter[2]).to.equal(1);
    })

    it('should add new row correctly', () => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()

        kNNRecommender.addNewRowToDataset(['user 4', 1, 0, 1, 1, 0, 0, 0])

        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()

        const userRecommendationsAfter = kNNRecommender.getAllRecommendationsForUserId('user 4')
        expect(userRecommendationsAfter[0]).to.equal('user 4');
        expect(userRecommendationsAfter[1]).to.equal(1);
        expect(userRecommendationsAfter[2]).to.equal(0);

    })


    it('should fail gracefully with empty', () => {
        expect(() => new KNNRecommender(emptyUserItemMatrix)).to.throw()
    })

    it('should fail gracefully when not initiated', () => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        expect(() => kNNRecommender.getNNearestNeighboursForUserId('user 1', 1)).to.throw()
    })

    it('should fail gracefully with malformatted data in user item matrix', () => {
        const kNNRecommender = new KNNRecommender(malformattedDataInUserItemMatrix)
        expect(() => kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()).to.throw();
    })

    it('should fail gracefully with first user id being a number ', () => {
        expect(() => new KNNRecommender(malformattedUserIdANumber)).to.throw()
    })

    it('should fail gracefully with other than first user id being a number', () => {
        const kNNRecommender = new KNNRecommender(malformattedSecondUserIdANumber)
        expect(() => kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()).to.throw();
    })

    it('should fail gracefully with malformatted item name', () => {
        expect(() => new KNNRecommender(malformattedItemName)).to.throw()
    })

    it('should fail gracefully with wrong sized user item matrix', () => {
        const kNNRecommender = new KNNRecommender(wrongSizeUserItemMatrix)
        expect(() => kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()).to.throw();
    })

    it('should fail gracefully with second wrong sized user item matrix', () => {
        const kNNRecommender = new KNNRecommender(wrongSize2UserItemMatrix)
        expect(() => kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()).to.throw();
    })
})