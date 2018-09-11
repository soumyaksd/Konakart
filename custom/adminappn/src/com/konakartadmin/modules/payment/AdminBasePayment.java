package com.konakartadmin.modules.payment;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.konakart.app.NameValue;
import com.konakartadmin.bl.AdminBaseMgr;

/**
 * Base class for sending payment information to the payment gateway
 * 
 */
public class AdminBasePayment extends AdminBaseMgr
{
    /** the log */
    private static Log log = LogFactory.getLog(AdminBasePayment.class);

    /**
     * Sends data to the payment gateway via a POST. Parameters are received from the
     * AdminPaymentDetails object.
     * 
     * @param pd
     * @return The response to the post
     * @throws IOException
     */
    public String postData(AdminPaymentDetails pd) throws IOException
    {
        URL url = new URL(pd.getRequestUrl());
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();

        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setDoInput(true);

        if (pd.getReferrer() != null && pd.getReferrer().length() > 1)
        {
            connection.setRequestProperty("Referer", pd.getReferrer());
        }

        if (pd.getContentType() != null && pd.getContentType().length() > 1)
        {
            connection.setRequestProperty("content-type", pd.getContentType());
        }

        customizeConnection(connection);

        PrintWriter out = new PrintWriter(connection.getOutputStream());

        String reqStr = getGatewayRequest(pd);

        if (log.isDebugEnabled())
        {
            log.debug("Post URL = " + pd.getRequestUrl() + "\n");
            log.debug("Post string = " + reqStr + "\n");

            if (pd.getParmList() != null)
            {
                for (Iterator<NameValue> iterator = pd.getParmList().iterator(); iterator.hasNext();)
                {
                    NameValue nv = iterator.next();

                    log.debug(nv.getName() + " = " + nv.getValue());
                }
            }
        }

        // Send the message
        out.print(reqStr);
        out.close();

        // Get back the response
        StringBuffer respSb = new StringBuffer();
        BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
        String line = in.readLine();

        while (line != null)
        {
            respSb.append(line);
            line = in.readLine();
        }

        in.close();

        return respSb.toString();
    }

    /**
     * Sends data to the payment gateway via a GET. Parameters are received from the
     * AdminPaymentDetails object.
     * 
     * @param pd
     * @return The response to the post
     * @throws IOException
     */
    public String getData(AdminPaymentDetails pd) throws IOException
    {
        // Construct data for GET
        String urlStr = pd.getRequestUrl();
        String requestStr = getGatewayRequest(pd);

        if (log.isDebugEnabled())
        {
            log.debug("GET URL = " + urlStr + requestStr);
        }
        URL url = new URL(urlStr + requestStr);

        // Send data
        URLConnection conn = url.openConnection();

        // Get the response
        BufferedReader rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
        StringBuffer sbReply = new StringBuffer();
        String line;
        while ((line = rd.readLine()) != null)
        {
            sbReply.append(line);
        }
        rd.close();
        String result = sbReply.toString();

        return result;
    }

    /**
     * This method can be specialized in the super class to customize the format of the request.
     * 
     * @param pd
     *            the PaymentDetails
     */
    protected String getGatewayRequest(AdminPaymentDetails pd)
    {
        /*
         * If the PaymentDetails object already contains the string that we need to send (e.g. XML)
         * then we go no further and just send it
         */
        if (pd.getSendData() != null)
        {
            if (log.isDebugEnabled())
            {
                log.debug("GatewayRequest = \n" + pd.getSendData());
            }
            return pd.getSendData();
        }

        // Create the message from the parameters in the PaymentDetails object
        StringBuffer sb = new StringBuffer();
        int i = 0;
        for (Iterator<NameValue> iterator = pd.getParmList().iterator(); iterator.hasNext();)
        {
            NameValue nv = iterator.next();
            if (i > 0)
            {
                sb.append("&");
            }
            sb.append(nv.getName());
            sb.append("=");
            sb.append(nv.getValue());
            i++;
        }

        if (log.isDebugEnabled())
        {
            log.debug("GatewayRequest = \n" + sb.toString());
        }

        return sb.toString();
    }

    /**
     * This method is normally specialized in the super class to customize the connection
     * 
     * @param connection
     */
    protected void customizeConnection(HttpURLConnection connection)
    {
    }

    /**
     * Add the NameValue pairs in the parameters array into a hashMap
     * @param parameters an array of NameValue objects
     * @return a HashMap of the parameters
     */
    protected HashMap<String, String> getParameterHash(NameValue[] parameters)
    {
        HashMap<String, String> params = new HashMap<String, String>();
        if (parameters != null)
        {
            for (NameValue nv : parameters)
            {
                params.put(nv.getName(), nv.getValue());
            }
        }
        
        if (log.isDebugEnabled())
        {
            String optionParameters = "Option Parameters:";
            if (parameters != null)
            {
                for (NameValue nv : parameters)
                {
                    optionParameters += "\n\t\t " + nv.getName() + " = " + nv.getValue();
                }
            }
            log.debug(optionParameters);
        }

        return params;
    }
    
    /**
     * Class used to pass around the payment details
     */
    public class AdminPaymentDetails
    {
        /** Contains the parameters that are sent to the payment gateway */
        private List<NameValue> parmList;

        /** In some cases we may be sending data such as XML rather than a list of parameters */
        private String sendData;

        /** The URL used for the POST or GET */
        private String requestUrl;

        /** Holds the referrer - sometimes used to set on HTTP posts */
        private String referrer;

        /** Holds the contentType - sometimes used to set on HTTP posts */
        private String contentType;

        /**
         * Constructor
         */
        public AdminPaymentDetails()
        {
        }

        /**
         * Returns a string containing the attributes of the AdminPaymentDetail object
         * 
         * @return A String representing the AdminPaymentDetails object
         */
        public String toString()
        {
            StringBuffer str = new StringBuffer();
            str.append("AdminPaymentDetail:\n");
            str.append("referrer        = ").append(getReferrer()).append("\n");
            str.append("requestUrl      = ").append(getRequestUrl()).append("\n");
            str.append("contentType     = ").append(getContentType()).append("\n");
            str.append("sendData        = ").append(getSendData()).append("\n");
            if (parmList != null)
            {
                for (Iterator<NameValue> iterator = parmList.iterator(); iterator.hasNext();)
                {
                    NameValue nv = iterator.next();
                    str.append(nv.toString());
                }
            }
            return (str.toString());
        }

        /**
         * @return the parmList
         */
        public List<NameValue> getParmList()
        {
            return parmList;
        }

        /**
         * @param parmList
         *            the parmList to set
         */
        public void setParmList(List<NameValue> parmList)
        {
            this.parmList = parmList;
        }

        /**
         * @return the requestUrl
         */
        public String getRequestUrl()
        {
            return requestUrl;
        }

        /**
         * @param requestUrl
         *            the requestUrl to set
         */
        public void setRequestUrl(String requestUrl)
        {
            this.requestUrl = requestUrl;
        }

        /**
         * @return the referrer
         */
        public String getReferrer()
        {
            return referrer;
        }

        /**
         * @param referrer
         *            the referrer to set
         */
        public void setReferrer(String referrer)
        {
            this.referrer = referrer;
        }

        /**
         * @return the sendData
         */
        public String getSendData()
        {
            return sendData;
        }

        /**
         * @param sendData
         *            the sendData to set
         */
        public void setSendData(String sendData)
        {
            this.sendData = sendData;
        }

        /**
         * @return the contentType
         */
        public String getContentType()
        {
            return contentType;
        }

        /**
         * @param contentType
         *            the contentType to set
         */
        public void setContentType(String contentType)
        {
            this.contentType = contentType;
        }

    }
}
