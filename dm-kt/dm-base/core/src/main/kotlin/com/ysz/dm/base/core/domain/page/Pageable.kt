package com.ysz.dm.base.core.domain.page


/**
 * Abstract interface for pagination information.
 * @author carl
 * @since 2023-02-19 9:30 PM
 */
interface Pageable {
    /**
     * returns whether current pageable contains pagination information
     */
    fun isPaged(): Boolean = true

    fun isUnpaged() = !isPaged()

    /**
     * returns the page to be returned .
     */
    fun getPageNumber(): Int

    /**
     * returns the number of items to be returned
     */
    fun getPageSize(): Int


    /**
     * returns the offset to be taken according to the underlying page and page size
     */
    fun getOffset(): Long


    /**
     * return the sorting params
     */
    fun getSort(): Sort


    /**
     * next page
     */
    fun next(): Pageable
    fun previousOrFirst(): Pageable
    fun first(): Pageable
    fun withPage(pageNumber: Int): Pageable

    fun hasPrevious(): Boolean

    companion object {

    }

}