package com.ysz.dm.base.core.domain.page

/**
 * @author carl
 * @since 2023-02-20 2:29 AM
 **/
abstract class AbstractPageRequest(
    val page: Int,
    val size: Int
) : Pageable {
    init {
        check(page >= 0) { "Page index must be more than zero" }
        check(size > 0) { "Page size must be more than zero" }
    }


    override fun pageNumber(): Int = page


    override fun pageSize(): Int = size

    override fun offset(): Long = page.toLong() * size.toLong()


    override fun previousOrFirst(): Pageable = if (hasPrevious()) previous() else first()

    override fun hasPrevious(): Boolean = page > 0
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is AbstractPageRequest) return false

        if (page != other.page) return false
        if (size != other.size) return false

        return true
    }

    override fun hashCode(): Int {
        var result = page
        result = 31 * result + size
        return result
    }
}

class PageRequest(
    page: Int,
    size: Int,
    val sort: Sort = Sort.UNSORTED
) : AbstractPageRequest(page, size) {
    override fun sort(): Sort = this.sort

    override fun next(): PageRequest = PageRequest(page + 1, size, sort)

    override fun previous(): PageRequest = if (page == 0) this else PageRequest(page - 1, size, sort)

    override fun first(): PageRequest = PageRequest(0, size, sort)

    override fun withPage(pageNumber: Int): PageRequest = PageRequest(pageNumber, size, sort)

    fun withSort(newSort: Sort) = PageRequest(page, size, newSort)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PageRequest) return false
        if (!super.equals(other)) return false

        if (sort != other.sort) return false

        return true
    }

    override fun hashCode(): Int {
        var result = super.hashCode()
        result = 31 * result + sort.hashCode()
        return result
    }


}