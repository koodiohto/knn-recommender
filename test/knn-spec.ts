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


describe('basic test', () => {
    it('should get the first similar user correctly', () => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()
        const userToOtherUsersArray = kNNRecommender.getXNearestNeighboursForUserId('user 1', 1)

        expect(userToOtherUsersArray[0].otherUserId).to.equal('user 2');
        expect(userToOtherUsersArray[0].similarity).to.equal(3 / 5);
    })

    it('should get the similarities for three users correctly', () => {
        const kNNRecommender = new KNNRecommender(threeUserItemMatrix)
        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()
        const user1ToOtherUsersArray = kNNRecommender.getXNearestNeighboursForUserId('user 1', 2)

        expect(user1ToOtherUsersArray[0].otherUserId).to.equal('user 2');
        expect(user1ToOtherUsersArray[0].similarity).to.equal(3 / 5);
        expect(user1ToOtherUsersArray[1].otherUserId).to.equal('user 3');
        expect(user1ToOtherUsersArray[1].similarity).to.equal(2 / 5);

        const user2ToOtherUsersArray = kNNRecommender.getXNearestNeighboursForUserId('user 2')

        expect(user2ToOtherUsersArray[0].otherUserId).to.equal('user 1');
        expect(user2ToOtherUsersArray[0].similarity).to.equal(3 / 5);
        expect(user2ToOtherUsersArray[1].otherUserId).to.equal('user 3');
        expect(user2ToOtherUsersArray[1].similarity).to.equal(2 / 5);

        const user3ToOtherUsersArray = kNNRecommender.getXNearestNeighboursForUserId('user 3')

        expect(user3ToOtherUsersArray[0].otherUserId).to.equal('user 1');
        expect(user3ToOtherUsersArray[0].similarity).to.equal(2 / 5);
        expect(user3ToOtherUsersArray[1].otherUserId).to.equal('user 2');
        expect(user3ToOtherUsersArray[1].similarity).to.equal(2 / 5);

    })
})