package com.konakart.apiexamples;

import org.apache.torque.util.Criteria;
import org.apache.torque.util.Criteria.Criterion;

import com.konakart.appif.DataDescriptorIf;
import com.konakart.appif.FetchProductOptionsIf;
import com.konakart.appif.KKEngIf;
import com.konakart.appif.ProductSearchIf;
import com.konakart.db.KKCriteria;
import com.konakart.bl.ProductMgr;
import com.konakart.blif.ProductMgrIf;
import com.konakart.om.BaseProductsDescriptionPeer;
import com.konakart.om.BaseProductsPeer;

/**
 * An example of how to customize the Product manager in order to modify the standard behavior. The
 * konakart.properties file must be edited so that the customized manager is used rather than the
 * standard one:
 * 
 * konakart.manager.ProductMgr = com.konakart.bl.MyProductMgr
 */
public class MyProductMgr extends ProductMgr implements ProductMgrIf
{
    /**
     * Constructor
     * 
     * @param eng
     * @throws Exception
     */
    public MyProductMgr(KKEngIf eng) throws Exception
    {
        super(eng);
    }

    /**
     * The following method is a customization of the manageDataDescriptor method in order to modify
     * the standard product search behavior. It contains multiple examples.
     * 
     * Note that from version 5.0.0.0 the search rules of the custom fields (i.e. whether wildcards
     * should be added) can now be set directly in the DataDescriptor object.
     * 
     * @param dataDesc
     * @param c
     * @param price
     */
    protected void manageDataDescriptor(DataDescriptorIf dataDesc, Criteria c, String price)
    {
        super.manageDataDescriptor(dataDesc, c, price);

        int example = 0;

        if (example == 0)
        {
            /*
             * Example that shows how to convert the custom1 constraint into a LIKE rather than an
             * EQUALS.
             */
            if (dataDesc.getCustom1() != null)
            {
                Criteria.Criterion custom1Crit = c.getNewCriterion(BaseProductsPeer.CUSTOM1, "%"
                        + dataDesc.getCustom1() + "%", Criteria.LIKE);
                c.remove(BaseProductsPeer.CUSTOM1);
                c.add(custom1Crit);
            }
        } else if (example == 1)
        {
            /*
             * Example that shows how to convert the custom1 criteria into a LIKE rather than an
             * EQUALS and to OR it with the criteria that searches for a string in the product name,
             * description, model etc. This allows you to find a product that has the string
             * "myString" in the name or description or in the custom1 field
             */
            Criteria.Criterion custom1Crit = null;
            if (dataDesc.getCustom1() != null)
            {
                custom1Crit = c.getNewCriterion(BaseProductsPeer.CUSTOM1,
                        "%" + dataDesc.getCustom1() + "%", Criteria.LIKE);
                c.remove(BaseProductsPeer.CUSTOM1);
            }

            /*
             * Get the criterion that searches in the product description
             */
            Criterion crit = c.getCriterion(BaseProductsDescriptionPeer.PRODUCTS_NAME);
            if (crit != null && custom1Crit != null)
            {
                // Remove the criterion
                c.remove(BaseProductsDescriptionPeer.PRODUCTS_NAME);
                // Add the criterion again, OR'ed with the custom1 criterion
                c.add(crit.or(custom1Crit));
            }
        }
    }

    /**
     * Method that can be used to customize the query string before being sent to the database.
     * 
     * @param sessionId
     *            The sessionId of the logged in user
     * @param apiCall
     *            Name of the Api call sending the query
     * @param queryString
     *            Query string that will be sent to the database
     * @param dataDesc
     *            DataDescriptor object
     * @param prodSearch
     *            ProductSearch object which may be null
     * @param languageId
     *            Language id
     * @param options
     * @return Returns the customized query string
     * @throws Exception
     */
    protected String beforeSendQuery(String sessionId, String apiCall, String queryString,
            DataDescriptorIf dataDesc, ProductSearchIf prodSearch, int languageId,
            FetchProductOptionsIf options) throws Exception
    {
        return queryString;
    }

    /**
     * Method that can be used to customize the KKCriteria object before it is used to create the
     * query string that is sent to the database.
     * 
     * @param sessionId
     *            The sessionId of the logged in user
     * @param apiCall
     *            Name of the Api call sending the query
     * @param criteria
     *            KKCriteria object that will be used to create the query string
     * @param dataDesc
     *            DataDescriptor object
     * @param prodSearch
     *            ProductSearch object which may be null
     * @param languageId
     *            Language id
     * @param options
     * @return Returns the customized Torque Criteria
     * @throws Exception
     */
    protected KKCriteria beforeSendCriteria(String sessionId, String apiCall, KKCriteria criteria,
            DataDescriptorIf dataDesc, ProductSearchIf prodSearch, int languageId,
            FetchProductOptionsIf options) throws Exception
    {
        return criteria;
    }
}
