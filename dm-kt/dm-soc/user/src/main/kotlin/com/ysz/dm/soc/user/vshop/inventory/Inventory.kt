package com.ysz.dm.soc.user.vshop.inventory

data class VipProps(
    val vipLevel: Int
)

data class InventoryProps(
    val vipProps: VipProps?
)

data class Inventory(
    val inventoryId: Long,
    val version: Long,
    val createAt: Long,
    val updateAt: Long,
    val inventoryName: String,
    val inventoryDesc: String?,
    val sellPrices: String,
    val normalPrices: String,
    val props: InventoryProps,
)
