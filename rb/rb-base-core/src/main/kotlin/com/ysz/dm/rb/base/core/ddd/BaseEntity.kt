package com.ysz.dm.rb.base.core.ddd

/**
 * @author carl
 * @create 2022-11-10 4:38 PM
 **/
interface BaseEntity<ID> {
    fun id(): ID
}