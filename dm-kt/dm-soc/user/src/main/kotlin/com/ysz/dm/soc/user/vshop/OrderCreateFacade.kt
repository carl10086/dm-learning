package com.ysz.dm.soc.user.vshop

data class OrderItemDto(
    val inventoryId: Long,
    val quantity: Int,
    val price: String,
    /*property version num, 乐观锁, 防止价格改变造成观感 不一致*/
    val version: Long
)


data class OrderCreateDto(
    val userId: Long,
    val orderItems: List<OrderItemDto>,
    /*可能支付的领域在其他的地方，但是也可以用*/
    val pspType: PaymentServiceProvider
)

enum class PaymentServiceProvider
data class PrepareOrderResultDto(
    val orderId: String,
    val paymentOrderId: String,
)

data class InventorySnapshot(
    val inventoryId: Long,
    val version: Long,
    val createAt: Long,
    val updateAt: Long,
    val inventoryName: String,
    val inventoryDesc: String?,
    val sellPrices: String,
    val normalPrices: String,
)

interface InventoryService {
    fun checkUserCanBuy(userId: Long, orderItems: List<OrderItemDto>): Boolean
}

interface PaymentService {
}

class OrderCreateFacade(
    private val inventoryService: InventoryService
) {

    fun orderCreate(cmd: OrderCreateDto) {
        /*1. maybe*/
        inventoryService.checkUserCanBuy(cmd.userId, cmd.orderItems)
        /*2. check user can user this payment type*/
        /*3. */
        /*4. Get Inventory Snapshot*/
        /*5. Prepare for OrderId, PaymentId */
        /*6. Sign For Payment and ...*/
        /*7. Create Order*/
        /*8. Create Payment*/
        /*9. Sync Finish */
    }


}