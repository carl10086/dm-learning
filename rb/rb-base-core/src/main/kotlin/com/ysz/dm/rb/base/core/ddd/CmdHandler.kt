package com.ysz.dm.rb.base.core.ddd

/**
 * base command handler
 * @author carl
 * @create 2022-11-10 6:56 PM
 */
interface CmdHandler<REQ, RESP> {
    fun handle(req: REQ): RESP
}