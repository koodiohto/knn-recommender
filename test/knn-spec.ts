import KNNRecommender from '../src/knn-recommender';
import { expect } from 'chai';

import { generateABigMatrix } from './bigMatrixGenerator'


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

const onlyItemsMatrix: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7']
]

const oneDimensionalMatrix: any[] =
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7']

const emptyUserItemMatrix: any[][] = []

const malformattedUserItemMatrixWithTwoSameUserIds: any[][] = [
    ['emptycorner', 'item 1', 'item 2', 'item 3', 'item 4',
        'item 5', 'item 6', 'item 7'],
    ['user 1', 1, -1, 0, 0, -1, 1, 0],
    ['user 1', 1, -1, 0, 1, -1, 0, 0]
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


describe('basic test', () => {
    it('should get the first similar user correctly', (done) => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeRecommender().then(() => {
            const userToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 1', 1)

            expect(userToOtherUsersArray[0].otherRowId).to.equal('user 2');
            expect(userToOtherUsersArray[0].similarity).to.equal(3 / 5);
            done()
        })
    })

    it('should get the first similar user correctly when only one user initialized', () => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeRecommenderForUserId('user 1')

        const userToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 1', 1)

        expect(userToOtherUsersArray[0].otherRowId).to.equal('user 2');
        expect(userToOtherUsersArray[0].similarity).to.equal(3 / 5);

        expect(() => kNNRecommender.getNNearestNeighboursForUserId('user 2', 1)).to.throw()
    })

    it('should work with step by step test from README.md', () => {
        const kNNRecommender = new KNNRecommender(null)
        kNNRecommender.addNewItemToDataset('item 1')
        kNNRecommender.addNewItemToDataset('item 2')
        kNNRecommender.addNewEmptyUserToDataset('user 1')
        kNNRecommender.addNewEmptyUserToDataset('user 2')

        kNNRecommender.addDislikeForUserToAnItem('user 1', 'item 1')
        kNNRecommender.addLikeForUserToAnItem('user 1', 'item 2')
        kNNRecommender.addDislikeForUserToAnItem('user 2', 'item 1')

        kNNRecommender.initializeRecommenderForUserId('user 2')

        const user2Recommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 2')

        expect(user2Recommendations[0].itemId).to.equal('item 2');

        kNNRecommender.addNewItemToDataset('item 3')
        kNNRecommender.addLikeForUserToAnItem('user 2', 'item 3')

        kNNRecommender.initializeRecommenderForUserId('user 1')

        const user1Recommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 1', { amountOfDesiredNewRecommendations: 1 })

        expect(user1Recommendations[0].itemId).to.equal('item 3');

        kNNRecommender.addNewItemToDataset('item 4')
        kNNRecommender.addLikeForUserToAnItem('user 2', 'item 4')
        kNNRecommender.initializeRecommenderForUserId('user 1')
        const user1Recommendations2 = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 1', { amountOfDesiredNewRecommendations: 2 })

        expect(user1Recommendations2[1].itemId).to.equal('item 4');
    })

    it('should work with step by step test for item to item-characteristics', () => {
        const kNNRecommender = new KNNRecommender(null)

        kNNRecommender.addNewItemCharacteristicToDataset('characteristic 1')
        kNNRecommender.addNewItemCharacteristicToDataset('characteristic 2')
        kNNRecommender.addNewEmptyItemAsRowToDataset('item 1')
        kNNRecommender.addNewEmptyItemAsRowToDataset('item 2')
        kNNRecommender.addNewEmptyItemAsRowToDataset('item 3')

        kNNRecommender.addCharacteristicForItem('item 1', 'characteristic 1')
        kNNRecommender.addCharacteristicForItem('item 1', 'characteristic 2')

        kNNRecommender.addCharacteristicForItem('item 2', 'characteristic 1')

        kNNRecommender.initializeRecommenderForItemId('item 2')

        const similarItemsForItem2 = kNNRecommender.getNNearestNeighboursForItemId('item 2', 1)

        //should print item 1
        expect(similarItemsForItem2[0].otherRowId).to.equal('item 1');

        kNNRecommender.addNewItemCharacteristicToDataset('characteristic 3')
        kNNRecommender.addCharacteristicForItem('item 2', 'characteristic 3')

        kNNRecommender.initializeRecommenderForItemId('item 1')

        const similarItemsForItem1 = kNNRecommender.getNNearestNeighboursForItemId('item 1', 1)

        //should print item 2
        expect(similarItemsForItem1[0].otherRowId).to.equal('item 2');

    })


    it('should initialize with an empty matrix when users added first', (done) => {
        const kNNRecommender = new KNNRecommender(null)
        kNNRecommender.addNewUserToDataset(['user 1'])
        kNNRecommender.addNewUserToDataset(['user 2'])
        kNNRecommender.addNewItemToDataset('item 1')
        kNNRecommender.addNewItemToDataset('item 2')

        kNNRecommender.initializeRecommender().then(() => {

            const userToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 1', 1)

            expect(userToOtherUsersArray[0].otherRowId).to.equal('user 2');
            expect(userToOtherUsersArray[0].similarity).to.equal(0);
            done()
        })
    })

    it('should initialize with an empty matrix when items added first', (done) => {
        const kNNRecommender = new KNNRecommender(null)
        kNNRecommender.addNewItemToDataset('item 1')
        kNNRecommender.addNewItemToDataset('item 2')
        kNNRecommender.addNewEmptyUserToDataset('user 1')
        kNNRecommender.addNewEmptyUserToDataset('user 2')

        kNNRecommender.initializeRecommender().then(() => {
            kNNRecommender.addLikeForUserToAnItem('user 1', 'item 1')
            kNNRecommender.addLikeForUserToAnItem('user 2', 'item 1')

            kNNRecommender.initializeRecommender().then(() => {

                const userToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 1', 1)

                expect(userToOtherUsersArray[0].otherRowId).to.equal('user 2');
                expect(userToOtherUsersArray[0].similarity).to.equal(1);
                done()
            })
        })
    })

    it('should initialize with items only matrix ', (done) => {
        const kNNRecommender = new KNNRecommender(onlyItemsMatrix)
        kNNRecommender.addNewEmptyUserToDataset('user 1')
        kNNRecommender.addNewEmptyUserToDataset('user 2')

        kNNRecommender.initializeRecommender().then(() => {

            const userToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 1', 1)

            expect(userToOtherUsersArray[0].otherRowId).to.equal('user 2');
            expect(userToOtherUsersArray[0].similarity).to.equal(0);
            done()
        })
    })

    it('should get the similarities for three users correctly', (done) => {
        const kNNRecommender = new KNNRecommender(threeUserItemMatrix)
        kNNRecommender.initializeRecommender().then(() => {
            const user1ToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 1', 2)

            expect(user1ToOtherUsersArray[0].otherRowId).to.equal('user 2');
            expect(user1ToOtherUsersArray[0].similarity).to.equal(3 / 5);
            expect(user1ToOtherUsersArray[1].otherRowId).to.equal('user 3');
            expect(user1ToOtherUsersArray[1].similarity).to.equal(2 / 5);

            const user2ToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 2')

            expect(user2ToOtherUsersArray[0].otherRowId).to.equal('user 1');
            expect(user2ToOtherUsersArray[0].similarity).to.equal(3 / 5);
            expect(user2ToOtherUsersArray[1].otherRowId).to.equal('user 3');
            expect(user2ToOtherUsersArray[1].similarity).to.equal(2 / 5);

            const user3ToOtherUsersArray = kNNRecommender.getNNearestNeighboursForUserId('user 3')

            expect(user3ToOtherUsersArray[0].otherRowId).to.equal('user 1');
            expect(user3ToOtherUsersArray[0].similarity).to.equal(2 / 5);
            expect(user3ToOtherUsersArray[1].otherRowId).to.equal('user 2');
            expect(user3ToOtherUsersArray[1].similarity).to.equal(2 / 5);
            done()
        })
    })

    it('should get all the recommendations for user correctly', (done) => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeRecommender().then(() => {
            const userRecommendations = kNNRecommender.getAllRecommendationsForUserId('user 2')

            expect(userRecommendations[0]).to.equal('user 2');
            expect(userRecommendations[1]).to.equal(1);
            expect(userRecommendations[2]).to.equal(-1);
            expect(userRecommendations[3]).to.equal(0);
            done()
        })

    })

    it('should get 3 new unique recommendations correctly for user', (done) => {
        const kNNRecommender = new KNNRecommender(threeUserItemMatrixForFindingRecommendations)
        kNNRecommender.initializeRecommender().then(() => {
            const userRecommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 3', { amountOfDesiredNewRecommendations: 3, amountOfDesiredNearestNeighboursToUse: 2 })

            expect(userRecommendations[0].itemId).to.equal('item 1');
            expect(userRecommendations[0].recommenderUserId).to.equal('user 2');
            expect(userRecommendations[0].similarityWithRecommender).to.equal(3 / 4);
            expect(userRecommendations[1].itemId).to.equal('item 6');
            expect(userRecommendations[1].recommenderUserId).to.equal('user 1');
            expect(userRecommendations[1].similarityWithRecommender).to.equal(2 / 5);
            expect(userRecommendations[2]).to.equal(undefined);

            const userRecommendations2 = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 2', { amountOfDesiredNewRecommendations: 40, amountOfDesiredNearestNeighboursToUse: 20 })
            expect(userRecommendations2[0].itemId).to.equal('item 6');
            expect(userRecommendations2[1]).to.equal(undefined);
            done()
        })
    })

    it('should get new recommendations correctly for user exluding some', (done) => {
        const kNNRecommender = new KNNRecommender(threeUserItemMatrixForFindingRecommendations)
        kNNRecommender.initializeRecommender().then(() => {
            const userRecommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 3', { amountOfDesiredNewRecommendations: 3, amountOfDesiredNearestNeighboursToUse: 2, excludingTheseItems: ['item 1'] })

            expect(userRecommendations[0].itemId).to.equal('item 6');
            expect(userRecommendations[0].recommenderUserId).to.equal('user 1');
            expect(userRecommendations[0].similarityWithRecommender).to.equal(2 / 5);
            expect(userRecommendations[1]).to.equal(undefined);

            const userRecommendations2 = kNNRecommender.generateNNewRecommendationsForUserId('user 3', { amountOfDesiredNewRecommendations: 3, amountOfDesiredNearestNeighboursToUse: 2, excludingTheseItems: ['item 1', 'item 6'] })
            expect(userRecommendations2[0]).to.equal(undefined);

            done()
        })
    })

    it('should get 3 new (not unique) recommendations correctly for user', (done) => {
        const kNNRecommender = new KNNRecommender(threeUserItemMatrixForFindingRecommendations)
        kNNRecommender.initializeRecommender().then(() => {
            const userRecommendations = kNNRecommender.generateNNewRecommendationsForUserId('user 3', { amountOfDesiredNewRecommendations: 3, amountOfDesiredNearestNeighboursToUse: 2 })

            expect(userRecommendations[0].itemId).to.equal('item 1');
            expect(userRecommendations[0].recommenderUserId).to.equal('user 2');
            expect(userRecommendations[0].similarityWithRecommender).to.equal(3 / 4);
            expect(userRecommendations[1].itemId).to.equal('item 1');
            expect(userRecommendations[1].recommenderUserId).to.equal('user 1');
            expect(userRecommendations[1].similarityWithRecommender).to.equal(2 / 5);
            expect(userRecommendations[2].itemId).to.equal('item 6');
            expect(userRecommendations[2].recommenderUserId).to.equal('user 1');
            expect(userRecommendations[2].similarityWithRecommender).to.equal(2 / 5);
            done()
        })
    })

    it('should get new recommendations correctly for over 10 sized user matrix', (done) => {
        const kNNRecommender = new KNNRecommender(generateABigMatrix(15, 15))
        kNNRecommender.initializeRecommender().then(() => {
            const userRecommendations = kNNRecommender.generateNNewRecommendationsForUserId('user 9', { amountOfDesiredNewRecommendations: 2, amountOfDesiredNearestNeighboursToUse: 5 })
            const userRecommendations2 = kNNRecommender.generateNNewRecommendationsForUserId('user 10', { amountOfDesiredNewRecommendations: 2, amountOfDesiredNearestNeighboursToUse: 5 })
            const userRecommendations3 = kNNRecommender.generateNNewRecommendationsForUserId('user 11', { amountOfDesiredNewRecommendations: 2, amountOfDesiredNearestNeighboursToUse: 5 })
            const userRecommendations4 = kNNRecommender.generateNNewRecommendationsForUserId('user 15', { amountOfDesiredNewRecommendations: 2, amountOfDesiredNearestNeighboursToUse: 5 })

            expect(userRecommendations[0].itemId).not.to.equal(undefined);
            expect(userRecommendations[0].recommenderUserId).not.to.equal(undefined);
            expect(userRecommendations[0].similarityWithRecommender).not.to.equal(undefined);

            expect(userRecommendations2[0].itemId).not.to.equal(undefined);
            expect(userRecommendations2[0].recommenderUserId).not.to.equal(undefined);
            expect(userRecommendations2[0].similarityWithRecommender).not.to.equal(undefined);

            expect(userRecommendations3[0].itemId).not.to.equal(undefined);
            expect(userRecommendations3[0].recommenderUserId).not.to.equal(undefined);
            expect(userRecommendations3[0].similarityWithRecommender).not.to.equal(undefined);

            expect(userRecommendations4[0].itemId).not.to.equal(undefined);
            expect(userRecommendations4[0].recommenderUserId).not.to.equal(undefined);
            expect(userRecommendations4[0].similarityWithRecommender).not.to.equal(undefined);

            done()
        }).catch((error) => {
            expect(false).equal(true)
            done()
        })
    })

    it('should update item for user correctly', (done) => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeRecommender().then(() => {

            const userRecommendationsBefore = kNNRecommender.getAllRecommendationsForUserId('user 1')
            expect(userRecommendationsBefore[0]).to.equal('user 1');
            expect(userRecommendationsBefore[2]).to.equal(-1);
            expect(userRecommendationsBefore[3]).to.equal(0);
            expect(userRecommendationsBefore[4]).to.equal(0);

            kNNRecommender.addLikeForUserToAnItem('user 1', 'item 2')
            kNNRecommender.addDislikeForUserToAnItem('user 1', 'item 3')
            kNNRecommender.updateMatrixForRowIdAndColumnId('user 1', 'item 4', -1)

            const userRecommendationsAfter = kNNRecommender.getAllRecommendationsForUserId('user 1')
            expect(userRecommendationsAfter[0]).to.equal('user 1');
            expect(userRecommendationsAfter[2]).to.equal(1);
            expect(userRecommendationsAfter[3]).to.equal(-1);
            expect(userRecommendationsAfter[4]).to.equal(-1);
            done()
        })


    })

    it('should add new row correctly', (done) => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeRecommender().then(() => {

            kNNRecommender.addNewUserToDataset(['user 3', 1, 0, 1, 1, 0, 0, 0])

            kNNRecommender.initializeRecommender().then(() => {

                const userRecommendationsAfter = kNNRecommender.getAllRecommendationsForUserId('user 3')
                expect(userRecommendationsAfter[0]).to.equal('user 3');
                expect(userRecommendationsAfter[1]).to.equal(1);
                expect(userRecommendationsAfter[2]).to.equal(0);

                const userRecommendations = kNNRecommender.generateNNewUniqueRecommendationsForUserId('user 3')
                expect(userRecommendations[0].itemId).to.equal('item 2');
                expect(() => kNNRecommender.addNewUserToDataset(['user 3', 1, 0, 1, 1, 0, 0, 0])).to.throw()
            })
            done()
        })
    })

    it('should add new item correctly', (done) => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        kNNRecommender.initializeRecommender().then(() => {

            const userRecommendationsBefore = kNNRecommender.getAllRecommendationsForUserId('user 2')
            expect(userRecommendationsBefore[8]).to.equal(undefined);

            kNNRecommender.addNewItemToDataset('item 8')

            const userRecommendationsAfter = kNNRecommender.getAllRecommendationsForUserId('user 2')
            expect(userRecommendationsAfter[0]).to.equal('user 2');
            expect(userRecommendationsAfter[8]).to.equal(0);
            done()
        })
    })

    it('should work with a big matrix', function (done) {
        this.timeout(0);//disable timeout
        const bigMatix = generateABigMatrix(500, 500)
        const kNNRecommender = new KNNRecommender(bigMatix)
        const timeStampBefore = new Date()
        kNNRecommender.initializeRecommender().then(() => {
            const timeStampAfter = new Date()
            const timeDif = (timeStampAfter.getTime() - timeStampBefore.getTime())

            console.log(`time to initialize the big matrix was: ${timeDif}`)
            expect(timeDif).to.be.least(1000);

            const timeStampBefore2 = new Date()
            const userRecommendations = kNNRecommender.generateNNewRecommendationsForUserId('user 50', { amountOfDesiredNewRecommendations: 2, amountOfDesiredNearestNeighboursToUse: 20 })
            const timeStampAfter2 = new Date()

            expect((timeStampAfter2.getTime() - timeStampBefore2.getTime())).to.be.lessThan(10);
            expect(userRecommendations[0].itemId).not.to.equal(undefined);
            expect(userRecommendations[0].recommenderUserId).not.to.equal(undefined);
            expect(userRecommendations[0].similarityWithRecommender).to.be.greaterThan(0.1)
            expect(userRecommendations[0].similarityWithRecommender).to.be.lessThan(0.9)
            done()
        })

    })

    it('should fail gracefully with empty', () => {
        expect(() => new KNNRecommender(emptyUserItemMatrix)).to.throw()
    })

    it('should fail gracefully with one dimensional matrix', () => {
        expect(() => new KNNRecommender(oneDimensionalMatrix)).to.throw()
    })


    it('should fail gracefully with two same user ids in the data', (done) => {
        const kNNRecommender = new KNNRecommender(malformattedUserItemMatrixWithTwoSameUserIds)

        kNNRecommender.initializeRecommender().then(() => {
            done(new Error('Expected method to reject.'))
        }).catch((err) => {
            expect(true)
            done();
        })
    })



    it('should fail gracefully when not initiated', () => {
        const kNNRecommender = new KNNRecommender(simpleUserItemMatrix)
        expect(() => kNNRecommender.getNNearestNeighboursForUserId('user 1', 1)).to.throw()
    })

    it('should fail gracefully with other than first user id being a number', (done) => {
        const kNNRecommender = new KNNRecommender(malformattedSecondUserIdANumber)
        kNNRecommender.initializeRecommender().then(() => {
            done(new Error('Expected method to reject.'))
        }).catch((err) => {
            expect(true)
            done();
        })
    })

    it('should fail gracefully with malformatted item name', () => {
        expect(() => new KNNRecommender(malformattedItemName)).to.throw()
    })

})