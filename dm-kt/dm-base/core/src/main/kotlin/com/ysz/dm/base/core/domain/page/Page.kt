package com.ysz.dm.base.core.domain.page

import kotlin.math.ceil

/**
 * A page is a sublist of a list of objects .
 * It allows gain information about the position of it in the containing
 * @author carl
 * @since 2023-02-19 9:46 PM
 **/
interface Page<T> {
    fun getContent(): List<T>

    fun getTotalElements(): Long
    fun <U> map(converter: (T) -> U): Page<U>
    fun hasNext(): Boolean
    fun isLast(): Boolean
    fun getTotalPages(): Int

    companion object {
        fun <T> empty(): Page<T> = PageImpl(emptyList(), 0L, Pageable.unpaged())
    }

}


data class PageImpl<T>(
    private val content: List<T>,
    private val total: Long,
    private val pageable: Pageable
) : Page<T> {
    override fun getContent(): List<T> = content

    override fun getTotalElements(): Long {
        return total
    }

    override fun <U> map(converter: (T) -> U): PageImpl<U> {
        return PageImpl(
            this.content.map(converter),
            total,
            pageable
        )
    }

    override fun hasNext(): Boolean {
        return getNumber() + 1 < getTotalPages()
    }

    override fun isLast(): Boolean {
        return !hasNext()
    }

    override fun getTotalPages(): Int =
        if (getSize() == 0) 1 else ceil(total.toDouble() / getSize().toDouble()).toInt()

    private fun getNumber(): Int {
        return if (pageable.paged()) pageable.pageNumber() else 0
    }

    private fun getSize(): Int {
        return if (pageable.paged()) pageable.pageSize() else content.size
    }


}