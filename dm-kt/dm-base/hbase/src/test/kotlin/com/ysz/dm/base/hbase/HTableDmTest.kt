package com.ysz.dm.base.hbase

import org.apache.hadoop.hbase.client.HTable
import org.apache.hadoop.hbase.client.Put
import org.apache.hadoop.hbase.client.Row
import org.apache.hadoop.hbase.util.Bytes
import org.slf4j.LoggerFactory

/**
 *
 * a example for row:
 *
 *     companion object {
 *
 *         private val fam = Bytes.toBytes("c")
 *         private val col = Bytes.toBytes("s")
 *
 *         private fun toRow(src: AtlasWlAct): Row {
 *             val rowKey = Bytes.toBytes(HBaseTools.normalize(src.atlasId.toString()))
 *
 *             return when (src.type) {
 *                 AtlasWlActType.del -> Delete(rowKey).apply {
 *                     addColumn(fam, col)
 *                 }
 *
 *                 else -> Put(rowKey).apply { addColumn(fam, col, ByteArray(1) { src.status }) }
 *             }
 *         }
 *     }
 *
 * @author carl.yu
 * @since 2023/1/3
 */
internal class HTableDmTest {

    /**
     * bulk save
     */
    private fun bulkSaveRows(rows: List<Row>, table: HTable): Array<Any?> {
        val results = Array<Any?>(rows.size) { null }
        table.batch(rows, results)
        log.info("success batch size:{}", rows.size)

        Put(Bytes.toBytes("aa")).apply {
//            addColumn(Bytes.toBytes("f"), Bytes.toBytes("c"), Bytes.toBytes)
        }
        return results
    }


    companion object {
        private val log = LoggerFactory.getLogger(HTableDmTest::class.java)


    }
}