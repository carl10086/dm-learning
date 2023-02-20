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
    fun paged(): Boolean = true

    fun unpaged() = !paged()

    /**
     * returns the page to be returned .
     */
    fun pageNumber(): Int

    /**
     * returns the number of items to be returned
     */
    fun pageSize(): Int


    /**
     * returns the offset to be taken according to the underlying page and page size
     */
    fun offset(): Long


    /**
     * return the sorting params
     */
    fun sort(): Sort


    /**
     * next page
     */
    fun next(): Pageable
    fun previousOrFirst(): Pageable
    fun previous(): Pageable
    fun first(): Pageable
    fun withPage(pageNumber: Int): Pageable

    fun hasPrevious(): Boolean

    companion object {
        fun unpaged(): Pageable = Unpaged.INSTANCE

    }
}


enum class Unpaged : Pageable {
    INSTANCE;

    override fun paged(): Boolean {
        return false
    }

    override fun unpaged(): Boolean {
        return true
    }

    override fun pageNumber(): Int {
        throw UnsupportedOperationException()
    }

    override fun pageSize(): Int {
        throw UnsupportedOperationException()
    }

    override fun offset(): Long {
        throw UnsupportedOperationException()
    }

    override fun sort(): Sort = Sort.UNSORTED

    override fun next(): Pageable = this

    override fun previousOrFirst(): Pageable = this

    override fun first(): Pageable = this

    override fun withPage(pageNumber: Int): Pageable = this

    override fun hasPrevious(): Boolean = false

    override fun previous(): Pageable = this
}