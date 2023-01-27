package com.ysz.dm.lib.lang.functions

/**
 * @author carl
 * @create 2022-11-18 11:49 AM
 **/
class ItemHolder<T> {

    private val items = mutableListOf<T>()

    fun addItem(x: T) {
        items.add(x)
    }


    fun getLastItem(): T? = items.lastOrNull()
}


/*func without class*/
fun <T> ItemHolder<T>.addAllItems(xs: List<T>) {
    xs.forEach { addItem(it) }
}

/**
 * @param builder a function as param. receiver is ItemHolder<T> , without parameters. you can use outside params like closure
 */
fun <T> itemHolderBuilder(builder: ItemHolder<T>.() -> Unit): ItemHolder<T> = ItemHolder<T>().apply { builder }