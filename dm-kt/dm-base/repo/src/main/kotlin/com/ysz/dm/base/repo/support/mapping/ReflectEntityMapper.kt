package com.ysz.dm.base.repo.support.mapping

import com.ysz.dm.base.repo.anotation.*
import java.sql.ResultSet
import kotlin.reflect.*
import kotlin.reflect.full.memberProperties
import kotlin.reflect.full.primaryConstructor
import kotlin.reflect.jvm.javaField
import kotlin.reflect.jvm.jvmErasure

/**
 * @author carl
 * @since 2023-02-18 1:48 AM
 **/
interface ReflectEntityMapper<T : Any> {
    /**
     * return all the values by the order of the domain class
     */
    fun propertyValues(entity: T, withPrimaryKey: Boolean = true, withVersionColumn: Boolean = true): List<Any?>

    /**
     * a rowMapper used for query
     */
    fun rowMapper(): (ResultSet) -> T

    /**
     * the property value of primary key
     */
    fun primaryKeyValue(entity: T): Any?

    fun changePrimaryKeyValue(entity: T, pkValue: Number)

    fun versionPropValue(entity: T): Any?

    fun initVersionPropValue(entity: T)

    fun domainMeta(): ReflectDomainMeta<T>
}


/**
 * reflect info for domain class .
 *
 * all data is unchanged for thread-safe
 */
data class ReflectDomainMeta<T : Any>(
    val kClass: KClass<T>,/*all columns have the same order of properties . the order is useful for performance*/
    val columns: List<String>,

    /*cache for col to Property*/
    val columnToPropertyMap: Map<String, String>,/*cache for property to Col*/
    val propertyToColumnMap: Map<String, String>,

    /*this domain class need auto generate id by database*/
    val autoGenerateId: Boolean,

    val primaryKeyColumn: String,

    val versionColumn: String? = null,
    val versionColumnDefaultValue: Number? = null,
) {
    //    fun columnToProperty(column: String): String = this.columnToPropertyMap[column]!!
    fun propertyToColumn(property: String): String = this.propertyToColumnMap[property]!!
    fun columnsWithoutPrimaryKey(): List<String> = this.columns.filter { it != primaryKeyColumn }

    fun hasVersion(): Boolean = this.versionColumn != null
}


class KotlinDataClassMapper<T : Any>(
    private val kClass: KClass<T>,
    converter: PropertyColNameConverter,
    private val propertyValueMapper: PropertyValueMapper,
) : ReflectEntityMapper<T> {

    /*primary key property cache*/
    private val _primaryProperty: KProperty1<T, *>

    private val _versionProperty: KMutableProperty1<T, *>?

    /*primary constructor*/
    private val _constructor: KFunction<T>
    private val _properties: List<KProperty1<T, *>>
    private val _parameters: List<KParameter>
    private val _domainMeta: ReflectDomainMeta<T>

    init {
        check(kClass.isData) { "$kClass is not a data class" }

        /*1. get all properties*/
        this._properties = kClass.memberProperties.filter {
            it.javaField!!.getAnnotation(Transient::class.java) == null
        }

        /*2. find primary key*/
        val primaryKeys =
            this._properties/*must be Id */.filter { it.javaField!!.getAnnotation(Id::class.java) != null }
        check(primaryKeys.isNotEmpty() && primaryKeys.size == 1) { "only One Id support" }
        this._primaryProperty = primaryKeys.first()

        /*3. find primary constructor and parameters*/
        this._constructor = kClass.primaryConstructor!!
        this._parameters = _constructor.parameters

        /*4. build meta*/
        val autoGenerateId = this._primaryProperty.javaField!!.getAnnotation(GeneratedValue::class.java) != null
        if (autoGenerateId) check(_primaryProperty is KMutableProperty1<T, *>) {
            "domainType must can change id when autoGenerateId is enabled, if kotlin, it means var instead of val"
        }

        val columnAnnoProperties = this._properties.filter { it.javaField!!.getAnnotation(Column::class.java) != null }

        val propertyToColumnMap = buildMap {
            for (name in _properties.asSequence().map { it.name }) {
                put(name, converter.propertyToColName(name))
            }

            for (annoProperty in columnAnnoProperties) {
                put(annoProperty.name, annoProperty.javaField!!.getAnnotation(Column::class.java).name)
            }
        }

        /*5. handle version column*/
        val versionColumns = this._properties.filter { it.javaField!!.getAnnotation(Version::class.java) != null }
            .apply { check(size <= 1) { "at most one version column" } }


        val versionProperty = if (versionColumns.isNotEmpty()) versionColumns.first() else null
        val versionColumnDefaultValue: Number? = versionProperty?.let {
            check(it is KMutableProperty1<T, *>) { "version column must can be changed , in kotlin, it means var instead of val" }
            when (it.returnType.jvmErasure) {
                Long::class -> 1L
                Int::class -> Integer.valueOf(1) /*toInt can't remove or something wrong ~*/
                else -> throw IllegalArgumentException("only support Long, int class for version column")
            }
        }
        this._versionProperty = versionProperty?.let { it as KMutableProperty1<T, *> }


        this._domainMeta =
            ReflectDomainMeta(
                this.kClass,
                this._properties.map { propertyToColumnMap[it.name]!! },
                buildMap {
                    propertyToColumnMap.forEach { (t, u) -> put(u, t) }
                },
                propertyToColumnMap,
                autoGenerateId,
                _primaryProperty.name,
                _versionProperty?.name,
                versionColumnDefaultValue
            )
    }


    override fun rowMapper(): (ResultSet) -> T {/*根据 name 去查询 constructor 对应的 KParameter*/
        val associateBy = this._parameters.associateBy { it.name!! }
        return {
            val list = buildMap {
                for (param in _parameters) {
                    if (_domainMeta.propertyToColumnMap.contains(param.name!!)) {
                        put(
                            associateBy[param.name!!]!!, propertyValueMapper.getValueFromResultSet(
                                it, _domainMeta.propertyToColumn(param.name!!), param.type.jvmErasure
                            )
                        )
                    }
                }
            }

            _constructor.callBy(list)
        }
    }

    override fun initVersionPropValue(entity: T) {
        this._versionProperty?.setter?.call(entity, this._domainMeta.versionColumnDefaultValue)
    }

    override fun versionPropValue(entity: T): Any? =
        this._versionProperty?.get(entity)

    override fun changePrimaryKeyValue(entity: T, pkValue: Number) {
        val prop = this._primaryProperty
        if (prop is KMutableProperty1<T, *>) {
            when (prop.setter.parameters[1].type.jvmErasure) {
                Int::class -> prop.setter.call(entity, pkValue.toInt())
                Long::class -> prop.setter.call(entity, pkValue.toLong())
                else -> prop.setter.call(entity, pkValue)
            }

        }

    }

    override fun domainMeta(): ReflectDomainMeta<T> = this._domainMeta


    override fun primaryKeyValue(entity: T): Any? = this._primaryProperty.get(entity)

    override fun propertyValues(entity: T, withPrimaryKey: Boolean, withVersionColumn: Boolean): List<Any?> =
        _properties.asSequence().filter {
            when {
                !withPrimaryKey && it.name == this._domainMeta.primaryKeyColumn -> false
                !withVersionColumn && it.name == this._domainMeta.versionColumn -> false
                else -> true
            }
        }.map { it.get(entity) }.toList()

}

