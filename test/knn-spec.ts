import KNNRecommender from '../src/knn-recommender';
import { expect } from 'chai';

const userItemMatrix: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, -1, 0, 0, -1, 1, 0],
    ['user 2', 1, -1, 0, 1, -1, 0, 0]
]


describe('basic test', () => {

    before(() => {

    });

    after(() => {
    });

    it('should get the first similar user correctly', () => {
        const kNNRecommender = new KNNRecommender(userItemMatrix)
        kNNRecommender.initializeKNNRecommenderForZeroOneUserMatrix()
        const userToOtherUsersArray = kNNRecommender.getXNearestNeighboursForUserId('user 1', 1)

        //console.log(`userItemMatrix.length: ${userItemMatrix.length}`)

        expect(userToOtherUsersArray[0].otherUserId).to.equal('user 2');
        expect(userToOtherUsersArray[0].similarity).to.equal(3 / 5);
    })
})