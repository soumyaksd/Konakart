package com.konakart.apiexamples;

import java.math.BigDecimal;

import org.apache.torque.TorqueException;

import com.konakart.app.KKException;
import com.konakart.appif.KKEngIf;
import com.konakart.bl.TaxMgr;
import com.konakart.blif.TaxMgrIf;
import com.workingdogs.village.DataSetException;

/**
 * An example of how to customize the Tax manager in order to modify the standard behavior. In this
 * case the tax methods have been customized to always return zero tax. The konakart.properties file
 * must be edited so that the customized manager is used rather than the standard one:
 * 
 * konakart.manager.TaxMgr = com.konakart.bl.MyTaxMgr
 * 
 */
public class MyTaxMgr extends TaxMgr implements TaxMgrIf
{

    /**
     * Constructor
     * 
     * @param eng
     * @throws Exception
     */
    public MyTaxMgr(KKEngIf eng) throws Exception
    {
        super(eng);
    }

    public BigDecimal addStoreTax(BigDecimal cost, int taxClassId) throws KKException,
            TorqueException, DataSetException
    {
        return cost;
    }

    public BigDecimal addStoreTax(BigDecimal cost, int taxClassId, int quantity)
            throws KKException, TorqueException, DataSetException
    {
        return cost;
    }

    public BigDecimal addTax(BigDecimal cost, int countryId, int zoneId, int taxClassId)
            throws KKException, TorqueException, DataSetException
    {
        return cost;
    }

    public BigDecimal addTax(BigDecimal cost, int countryId, int zoneId, int taxClassId,
            int quantity) throws KKException, TorqueException, DataSetException
    {
        return cost;
    }

    public BigDecimal getTax(BigDecimal cost, int countryId, int zoneId, int taxClassId)
            throws KKException, TorqueException, DataSetException
    {
        return new BigDecimal(0);
    }

    public BigDecimal getTax(BigDecimal cost, int countryId, int zoneId, int taxClassId,
            int quantity) throws KKException, TorqueException, DataSetException
    {
        return new BigDecimal(0);
    }
}
