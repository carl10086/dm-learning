package com.ysz.dm.base.core.domain.page

/**
 * @author carl
 * @since 2023-02-19 9:37 PM
 **/
data class Sort(
    val orders: List<Order>
) {
    constructor(direction: Direction, vararg props: String) : this(
        props.map { Order(direction, it) }
    )
}

data class Order(
    val direction: Direction,
    val prop: String
)

enum class Direction {
    ASC, DESC;
}