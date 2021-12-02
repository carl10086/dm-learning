package com.ysz.dm.fast.algorithm.sort;

import java.util.Arrays;

public class Solution {

  private java.util.Random random = new java.util.Random();

  public int[] MySort(int[] arr) {
    if (arr == null) {
      return null;
    }
    if (arr.length == 0) {
      return arr;
    }

    fastSort(arr, 0, arr.length - 1);
    return arr;
  }


  private void fastSort(int[] array, int left, int right) {
    int childArrayLen = right - left + 1;
    if (childArrayLen == 1) {
      return;
    }

    if (childArrayLen == 2) {
      if (array[left] > array[right]) {
        swap(array, left, right);
      }

      return;
    }

    final int randomIndex = doFastSort(array, left, right);
    // 0 -> randomIndex
    fastSort(array, left, randomIndex);
    // >= randomIndx <= right
    fastSort(array, randomIndex, right);
  }

  private int doFastSort(int[] array, int left, int right) {
    int childArrayLen = right - left + 1;
    if (childArrayLen == 1) {
      return 0;
    }

    if (childArrayLen == 2) {
      if (array[left] > array[right]) {
        swap(array, left, right);
        return 1;
      }
    }

    swapRandom(array, left, right);

    int ranDomVal = array[right];

    int i = left;

    int j = right - 1;

    int randomIndx = -1;

    for (; ; ) {
      if (i == j) {
        /*和 ranDown 交换*/
        randomIndx = i;
        break;
      }

      /*寻找第一个比 randomVal 要大的吧*/
      int biggerIdx = -1;
      for (int k = i; k <= j; k++) {
        if (array[k] > ranDomVal) {
          biggerIdx = k;
          break;
        }
      }

      if (biggerIdx == -1) {
        i = j;
        randomIndx = i;
        break;
      } else {
        i = biggerIdx;
      }


      /*2. 寻找第二个 <=randomVal 的值*/
      int smallerIndex = -1;

      /*2.1 找到了*/
      for (int k = j; k >= i; k--) {
        if (array[k] <= ranDomVal) {
          smallerIndex = k;
          break;
        }
      }

      /*2.2  找不到*/

      if (smallerIndex == -1) {
        j = i;
        randomIndx = i;
        break;
      } else {
        j = smallerIndex;
      }

      swap(array, biggerIdx, smallerIndex);

    }

    /*交换写到最后*/
    if (array[randomIndx] > ranDomVal) {
      swap(array, randomIndx, right);
      return randomIndx;
    } else {
      return right;
    }
  }


  private void swapRandom(int[] array, int left, int right) {
    int childArrayLen = right - left + 1;
    //0 -> childArrayLen-1
    int randomInx = random.nextInt(childArrayLen) + left;
//    int randomInx = 4;
    swap(array, randomInx, right);
  }


  private void swap(int[] array, int x, int y) {
    /*TODO 可以用位运算代替,少个 tmep? */

    int temp = array[x];
    array[x] = array[y];
    array[y] = temp;
  }


  public static void main(String[] args) throws Exception {
    int[] array = new int[]{-1, 2, 5, 7, 1, 4, 6, 3, 8, 10};
    new Solution().MySort(array);
    System.out.println(Arrays.toString(array));
  }

}
