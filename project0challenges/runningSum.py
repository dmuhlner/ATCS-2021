nums = [1, 2, 3, 4]


def runningSum(nums):
    sumNums =[]
    for index, element in enumerate(nums):
        if index == 0:
            sumNums.append(nums[0])
        else:
            sumNums.append(sumNums[index - 1] + nums[index])
    return sumNums

print(runningSum(nums))
nums = [1, 1, 1, 1]
print(runningSum(nums))
nums = [3, 1, 2, 10, 1]
print(runningSum(nums))