export const generateABigMatrix = (ITEM_SIZE: number = 500, USER_SIZE: number = 500) => {

    let bigMatrix: any[][] = new Array(USER_SIZE)

    const getRandomInt = (min: number, max: number) => {
        min = Math.ceil(min);
        max = Math.floor(max);
        return Math.floor(Math.random() * (max - min) + min);
    }

    for (let i = 0; i <= USER_SIZE; i++) {
        bigMatrix[i] = new Array(ITEM_SIZE)
        for (let j = 0; j <= ITEM_SIZE; j++) {
            let value: string | number = getRandomInt(-1, 2)
            if (i === 0 && j === 0) {
                value = 'emptycorner'
            } else if (j === 0) {
                value = `user ${i}`
            } else if (i === 0) {
                value = `item ${j}`
            }

            bigMatrix[i][j] = value
        }
    }
    //console.log("bigmatrix: " + JSON.stringify(bigMatrix))
    return bigMatrix
}