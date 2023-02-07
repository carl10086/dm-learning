package com.ysz.dm.lib.leecode

import java.util.LinkedList


/**
 *<pre>
 * https://leetcode.cn/problems/remove-sub-folders-from-the-filesystem/
 * class desc here
 *
 * 这个代码不合适. 可以优化为 和 前 1 个比较, 属于脑筋急转弯. 看排序后的结果
 *</pre>
 *@author carl.yu
 *@since 2023/2/8
 **/
class LeeCode1233 {}

internal object Solution {

    fun removeSubfolders(folder: Array<String>): List<String> {

        val sorted = folder.sorted()
        val result = ArrayList<String>()
        result.add(sorted[0])

        var current: String
        var prev: String = sorted[0]
        for (i in 1 until sorted.size) {
            current = sorted[i]
            if (!current.startsWith(prev + "/")) {
                result.add(current)
                prev = current
            }
        }


        return result
    }
}

fun main(args: Array<String>) {
    println(
        Solution.removeSubfolders(
            arrayOf(
                "/a/b/c", "/a/b/ca", "/a/b/d"
            )
        )
    )
}